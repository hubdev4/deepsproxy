/*
 * File: chat.ts
 * Project: deepsproxy
 * Author: Pedro Farias
 * Created: 2026-05-09
 * 
 * Last Modified: Sat May 09 2026
 * Modified By: Pedro Farias
 */

import { Context } from 'hono';
import { stream as honoStream } from 'hono/streaming';
import { v4 as uuidv4 } from 'uuid';
import { createDeepSeekStream, updateSessionParent } from '../services/deepseek.ts';
import { OpenAIRequest, ChoiceDelta, Message } from '../utils/types.ts';
import { robustParseJSON } from '../utils/json.ts';
import { registry } from '../tools/registry.ts';
import type { FunctionToolDefinition } from '../tools/types.ts';

type ParsedPlainCallingTool = {
  leadIn: string;
  name: string;
  arguments: Record<string, unknown>;
};

function getAllowedToolNames(tools: any): Set<string> {
  if (!Array.isArray(tools)) return new Set();
  return new Set(tools
    .map((tool: any) => tool?.type === 'function' ? tool?.function?.name : tool?.name)
    .filter((name: any) => typeof name === 'string' && name.length > 0));
}

function parsePlainCallingTool(text: string, allowedToolNames: Set<string>): ParsedPlainCallingTool | null {
  const match = text.match(/(^|\n)Calling:\s*([A-Za-z0-9_-]+)\s*\n\n([\s\S]+)$/);
  if (!match) return null;

  const name = match[2];
  if (allowedToolNames.size > 0 && !allowedToolNames.has(name)) return null;

  const parsedArgs = robustParseJSON(match[3].trim());
  if (!parsedArgs || typeof parsedArgs !== 'object' || Array.isArray(parsedArgs)) return null;

  const markerIndex = match.index! + match[1].length;
  return {
    leadIn: text.substring(0, markerIndex),
    name,
    arguments: parsedArgs as Record<string, unknown>
  };
}

function makeToolCallDelta(toolName: string, toolArgs: Record<string, unknown>, index?: number) {
  return {
    index,
    id: 'call_' + uuidv4(),
    type: 'function',
    function: {
      name: toolName,
      arguments: JSON.stringify(toolArgs)
    }
  };
}

function stringifyMessageContent(content: any): string {
  if (Array.isArray(content)) {
    return content.map((part: any) => {
      if (typeof part === 'string') return part;
      if (part && typeof part === 'object' && typeof part.text === 'string') return part.text;
      return JSON.stringify(part);
    }).join('\n');
  }

  if (content && typeof content === 'object') return JSON.stringify(content);
  return content || '';
}

function serializeToolCall(tc: any): string {
  let args = tc.function?.arguments || '{}';
  if (typeof args !== 'string') args = JSON.stringify(args);
  return `<tool_call>{"name": "${tc.function?.name}", "arguments": ${args}}</tool_call>`;
}

async function collectNonStreamingCompletion(
  stream: ReadableStream,
  uiSessionId: string,
  completionId: string,
  model: string,
  promptTokens: number,
  allowedToolNames: Set<string>
) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();

  let currentAppendPath = '';
  let currentFragmentType = '';
  let reasoningContent = '';
  let content = '';
  let completionTokens = 0;
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.startsWith('data: ')) continue;

      const dataStr = trimmed.slice(6);
      if (dataStr === '[DONE]') continue;

      try {
        const chunk = JSON.parse(dataStr);
        let dsMessageId: any = null;
        if (chunk.response_message_id) {
          dsMessageId = chunk.response_message_id;
        } else if (chunk.v && typeof chunk.v === 'object') {
          if (chunk.v.response && chunk.v.response.message_id) {
            dsMessageId = chunk.v.response.message_id;
          } else if (chunk.v.message_id) {
            dsMessageId = chunk.v.message_id;
          }
        } else if (chunk.message_id) {
          dsMessageId = chunk.message_id;
        }

        if (dsMessageId) {
          updateSessionParent(uiSessionId, dsMessageId);
        }

        let vStr = '';
        let foundStr = false;

        if (typeof chunk.p === 'string') {
          currentAppendPath = chunk.p;
          if (chunk.p === 'response/accumulated_token_usage' && typeof chunk.v === 'number') {
            completionTokens = chunk.v;
          }
        }

        if (typeof chunk.v === 'string') {
          vStr = chunk.v;
          foundStr = true;
        } else if (chunk.v && typeof chunk.v === 'object') {
          if (chunk.v.response && chunk.v.response.fragments && chunk.v.response.fragments.length > 0) {
            const frag = chunk.v.response.fragments[0];
            if (typeof frag.content === 'string') {
              vStr = frag.content;
              foundStr = true;
              currentAppendPath = frag.type === 'THINK' ? 'response/thinking_content' : 'response/content';
              currentFragmentType = frag.type || '';
            }
          } else if (Array.isArray(chunk.v) && chunk.v.length > 0) {
            const firstObj = chunk.v[0];
            if (typeof firstObj.content === 'string') {
              vStr = firstObj.content;
              foundStr = true;
              currentAppendPath = firstObj.type === 'THINK' ? 'response/thinking_content' : 'response/content';
              currentFragmentType = firstObj.type || '';
            }
          }
        }

        if (chunk.p === 'response/fragments' && Array.isArray(chunk.v)) {
          const lastFrag = chunk.v[chunk.v.length - 1];
          if (lastFrag && lastFrag.type) {
            currentFragmentType = lastFrag.type;
          }
        }

        const isThinkingChunk = currentAppendPath.includes('thinking_content') ||
          currentAppendPath.includes('THINK') ||
          (currentAppendPath.includes('fragments/-1/content') && currentFragmentType === 'THINK');

        if (foundStr && vStr !== '' && vStr !== 'FINISHED') {
          if (isThinkingChunk) {
            reasoningContent += vStr;
          } else {
            content += vStr;
          }
        }
      } catch (e) {
        // Ignore malformed upstream SSE lines and continue collecting.
      }
    }
  }

  const toolCalls: any[] = [];
  const plainCallingTool = parsePlainCallingTool(content, allowedToolNames);
  if (plainCallingTool) {
    content = plainCallingTool.leadIn;
    toolCalls.push(makeToolCallDelta(plainCallingTool.name, plainCallingTool.arguments));
  }

  const toolCallRegex = /<tool_call>([\s\S]*?)<\/tool_call>/g;
  let match: RegExpExecArray | null;
  while ((match = toolCallRegex.exec(content)) !== null) {
    const toolJsonStr = match[1].trim();
    const toolCallObj = robustParseJSON(toolJsonStr);
    if (!toolCallObj) continue;

    const nameMatch = toolJsonStr.match(/<tool_call\s+name="([^"]+)"/);
    const toolName = nameMatch ? nameMatch[1] : toolCallObj.name || '';
    let toolArgs: Record<string, unknown> = {};
    if (toolCallObj.arguments && typeof toolCallObj.arguments === 'object') {
      toolArgs = toolCallObj.arguments;
    } else {
      for (const key of Object.keys(toolCallObj).filter(k => k !== 'name')) {
        toolArgs[key] = toolCallObj[key];
      }
    }

    toolCalls.push({
      id: 'call_' + uuidv4(),
      type: 'function',
      function: {
        name: toolName,
        arguments: JSON.stringify(toolArgs)
      }
    });
  }

  const strippedContent = content.replace(toolCallRegex, '').trim();
  const hasToolCalls = toolCalls.length > 0;
  const usage = {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: promptTokens + completionTokens,
    prompt_tokens_details: {
      cached_tokens: 0
    }
  };

  const message: any = {
    role: 'assistant',
    content: hasToolCalls ? (strippedContent || null) : content
  };
  if (reasoningContent) {
    message.reasoning_content = reasoningContent;
  }
  if (hasToolCalls) {
    message.tool_calls = toolCalls;
  }

  return {
    id: completionId,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{
      index: 0,
      message,
      logprobs: null,
      finish_reason: hasToolCalls ? 'tool_calls' : 'stop'
    }],
    usage
  };
}

export async function chatCompletions(c: Context) {
  try {
    const body: OpenAIRequest = await c.req.json();
    const isStream = body.stream ?? false;
    
    // Extract the full OpenAI conversation, not just the final message.
    // Tool continuations arrive as: user -> assistant.tool_calls -> tool.
    // If only the final tool message is forwarded, DeepSeek receives an orphaned
    // tool result and often returns an empty continuation.
    let prompt = '';
    const messages = body.messages || [];
    let systemPrompt = '';
    const toolCallNamesById = new Map<string, string>();
    
    for (const msg of messages) {
      const contentStr = stringifyMessageContent((msg as any).content);

      if (msg.role === 'system') {
        systemPrompt += contentStr + '\n\n';
      } else if (msg.role === 'user') {
        prompt += `User: ${contentStr}\n\n`;
      } else if (msg.role === 'assistant') {
        let assistantContent = contentStr;
        if ((msg as any).reasoning_content) {
          assistantContent = `<think>\n${(msg as any).reasoning_content}\n</think>\n${assistantContent}`;
        }
        if (msg.tool_calls && Array.isArray(msg.tool_calls)) {
          for (const tc of msg.tool_calls) {
            if (tc?.id && tc?.function?.name) {
              toolCallNamesById.set(tc.id, tc.function.name);
            }
            assistantContent += `\n${serializeToolCall(tc)}`;
          }
        }
        prompt += `Assistant: ${assistantContent.trim()}\n\n`;
      } else if (msg.role === 'tool' || msg.role === 'function') {
        const toolCallId = (msg as any).tool_call_id;
        const toolName = msg.name || (toolCallId ? toolCallNamesById.get(toolCallId) : undefined) || 'tool';
        const toolLabel = toolCallId ? `${toolName}, id=${toolCallId}` : toolName;
        prompt += `Tool Response (${toolLabel}): ${contentStr}\n\n`;
      }
    }

    // Inject tools instructions
    const bodyAny = body as any;
    const allowedToolNames = getAllowedToolNames(bodyAny.tools);
    if (bodyAny.tools && Array.isArray(bodyAny.tools) && bodyAny.tools.length > 0) {
      // Better formatting for tools
      const formattedTools = bodyAny.tools.map((t: any) => {
        if (t.type === 'function') {
          return {
            name: t.function.name,
            description: t.function.description || '',
            parameters: t.function.parameters
          };
        }
        return t;
      });
      const toolsJson = JSON.stringify(formattedTools, null, 2);
      
      systemPrompt += `\n\n# TOOLS AVAILABLE\nYou have access to the following tools:\n${toolsJson}\n\nTo use a tool, you MUST output a JSON object wrapped EXACTLY in these tags:\n<tool_call>\n{"name": "tool_name", "arguments": {"param_name": "value"}}\n</tool_call>\n\nRULES:\n1. You can call multiple tools by outputting multiple <tool_call> blocks consecutively.\n2. Do NOT output any other text after your <tool_call> blocks. Wait for the user to provide the tool response.\n3. The JSON must be valid and accurately follow the tool's parameters.\n\n`;
      
      if (bodyAny.tool_choice && typeof bodyAny.tool_choice === 'object' && bodyAny.tool_choice.function) {
        const forcedTool = bodyAny.tool_choice.function.name;
        systemPrompt += `CRITICAL: You MUST call the tool "${forcedTool}" in this response.\n\n`;
      }
    }

    const finalPrompt = systemPrompt ? `${systemPrompt}\n${prompt}` : prompt;

    const isThinkingModel = body.model.includes('thinking');
    const isProModel = body.model.includes('pro');

    // A session is new if it doesn't have any assistant messages yet.
    // This handles cases where the first request has [System, User] messages.
    const isNewSession = !messages.some(m => m.role === 'assistant');

    // Empty response retry logic
    let stream: ReadableStream;
    let uiSessionId = '';
    let retries = 3;
    while (retries > 0) {
      try {
        // If it's a new session, force parent_message_id to null
        const result = await createDeepSeekStream(finalPrompt, isThinkingModel, isProModel, isNewSession ? null : undefined);
        stream = result.stream;
        uiSessionId = result.uiSessionId;
        break; // Success
      } catch (err: any) {
        retries--;
        if (retries === 0) throw err;
        // Wait a bit before retrying
        await new Promise(r => setTimeout(r, 1000));
      }
    }

    const completionId = 'chatcmpl-' + uuidv4();
    const promptTokens = Math.ceil(finalPrompt.length / 3.5);

    if (!isStream) {
      const completion = await collectNonStreamingCompletion(stream!, uiSessionId, completionId, body.model, promptTokens, allowedToolNames);
      return c.json(completion);
    }

    c.header('Content-Type', 'text/event-stream');
    c.header('Cache-Control', 'no-cache');
    c.header('Connection', 'keep-alive');

    return honoStream(c, async (streamWriter: any) => {
      const writeEvent = async (data: any) => {
        await streamWriter.write(`data: ${JSON.stringify(data)}\n\n`);
      };

      const makeChoice = (delta: any, finishReason: string | null = null) => ({
        index: 0,
        delta,
        logprobs: null,
        finish_reason: finishReason
      });

      // Send initial chunk
      await writeEvent({
        id: completionId,
        object: 'chat.completion.chunk',
        created: Math.floor(Date.now() / 1000),
        model: body.model,
        choices: [makeChoice({ role: 'assistant', content: '' })]
      });

      const reader = stream.getReader();
      const decoder = new TextDecoder();
      
      let inThinkingState = false;
      let thinkingFragments: Record<string, boolean> = {};
      let currentFragIndex = 0;
      let currentAppendPath = '';
      let currentFragmentType = '';
      
      let reasoningBuffer = '';
      let contentEmitBuffer = '';
      let insideTool = false;
      let emittedToolCallCount = 0;
      const TOOL_START = '<tool_call>';
      const TOOL_END = '</tool_call>';

      let buffer = '';
      let completionTokens = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith('data: ')) continue;
          
          const dataStr = trimmed.slice(6);
          if (dataStr === '[DONE]') {
            await streamWriter.write('data: [DONE]\n\n');
            continue;
          }

          try {
            const chunk = JSON.parse(dataStr);
            let dsMessageId: any = null;
            if (chunk.response_message_id) {
              dsMessageId = chunk.response_message_id;
            } else if (chunk.v && typeof chunk.v === 'object') {
              if (chunk.v.response && chunk.v.response.message_id) {
                dsMessageId = chunk.v.response.message_id;
              } else if (chunk.v.message_id) {
                dsMessageId = chunk.v.message_id;
              }
            } else if (chunk.message_id) {
              dsMessageId = chunk.message_id;
            }

            if (dsMessageId) {
              updateSessionParent(uiSessionId, dsMessageId);
            }

            let vStr = '';
            let foundStr = false;
            let isThinkingChunk = false;

            if (typeof chunk.p === 'string') {
              currentAppendPath = chunk.p;
              if (chunk.p === 'response/accumulated_token_usage' && typeof chunk.v === 'number') {
                completionTokens = chunk.v;
              }
            }

            // Extract string value
            if (typeof chunk.v === 'string') {
              vStr = chunk.v;
              foundStr = true;
            } else if (chunk.v && typeof chunk.v === 'object') {
              // Handle old fragments format if it ever occurs
              if (chunk.v.response && chunk.v.response.fragments && chunk.v.response.fragments.length > 0) {
                const frag = chunk.v.response.fragments[0];
                if (typeof frag.content === 'string') {
                  vStr = frag.content;
                  foundStr = true;
                  currentAppendPath = frag.type === 'THINK' ? 'response/thinking_content' : 'response/content';
                  currentFragmentType = frag.type || '';
                }
              } else if (Array.isArray(chunk.v) && chunk.v.length > 0) {
                const firstObj = chunk.v[0];
                if (typeof firstObj.content === 'string') {
                  vStr = firstObj.content;
                  foundStr = true;
                  currentAppendPath = firstObj.type === 'THINK' ? 'response/thinking_content' : 'response/content';
                  currentFragmentType = firstObj.type || '';
                }
              }
            }

            // Detect fragment type changes - for v2.0.0, track which fragment is active
            if (chunk.p === 'response/fragments' && Array.isArray(chunk.v)) {
              const lastFrag = chunk.v[chunk.v.length - 1];
              if (lastFrag && lastFrag.type) {
                currentFragmentType = lastFrag.type;
              }
            }

            // Determine if it's thinking based on the current path OR fragment type (for v2.0.0)
            if (currentAppendPath.includes('thinking_content') ||
                currentAppendPath.includes('THINK') ||
                (currentAppendPath.includes('fragments/-1/content') && currentFragmentType === 'THINK')) {
              isThinkingChunk = true;
            }

            if (foundStr && vStr !== '') {
              if (vStr === 'FINISHED') continue;

              const delta: ChoiceDelta = {};

              // Map chunk to either reasoning_content or content
              if (isThinkingChunk) {
                inThinkingState = true;
                reasoningBuffer += vStr;
                delta.reasoning_content = vStr;

                await writeEvent({
                  id: completionId,
                  object: 'chat.completion.chunk',
                  created: Math.floor(Date.now() / 1000),
                  model: body.model,
                  choices: [makeChoice(delta)]
                });
              } else {
                inThinkingState = false;
                
                contentEmitBuffer += vStr;

                while (contentEmitBuffer.length > 0) {
                  if (!insideTool) {
                    const plainCallingTool = parsePlainCallingTool(contentEmitBuffer, allowedToolNames);
                    if (plainCallingTool) {
                      if (plainCallingTool.leadIn && emittedToolCallCount === 0) {
                        await writeEvent({
                          id: completionId,
                          object: 'chat.completion.chunk',
                          created: Math.floor(Date.now() / 1000),
                          model: body.model,
                          choices: [makeChoice({ content: plainCallingTool.leadIn })]
                        });
                      }

                      await writeEvent({
                        id: completionId,
                        object: 'chat.completion.chunk',
                        created: Math.floor(Date.now() / 1000),
                        model: body.model,
                        choices: [makeChoice({
                          tool_calls: [makeToolCallDelta(plainCallingTool.name, plainCallingTool.arguments, emittedToolCallCount)]
                        })]
                      });
                      emittedToolCallCount++;
                      contentEmitBuffer = '';
                      break;
                    }

                    const startIdx = contentEmitBuffer.indexOf(TOOL_START);
                    if (startIdx !== -1) {
                      // Found tool start. Emit everything before it as text
                      const textToEmit = contentEmitBuffer.substring(0, startIdx);
                      if (textToEmit && emittedToolCallCount === 0) {
                        await writeEvent({
                          id: completionId,
                          object: 'chat.completion.chunk',
                          created: Math.floor(Date.now() / 1000),
                          model: body.model,
                          choices: [makeChoice({ content: textToEmit })]
                        });
                      }
                      insideTool = true;
                      contentEmitBuffer = contentEmitBuffer.substring(startIdx + TOOL_START.length);
                      continue; // re-evaluate loop for tool end
                    } else {
                      // No full start tag. Check for partial match at the end
                      let flushIndex = contentEmitBuffer.length;
                      for (let i = 1; i <= TOOL_START.length; i++) {
                        if (contentEmitBuffer.endsWith(TOOL_START.substring(0, i))) {
                          flushIndex = contentEmitBuffer.length - i;
                          break;
                        }
                      }
                      
                      const textToEmit = contentEmitBuffer.substring(0, flushIndex);
                      if (textToEmit && emittedToolCallCount === 0) {
                        await writeEvent({
                          id: completionId,
                          object: 'chat.completion.chunk',
                          created: Math.floor(Date.now() / 1000),
                          model: body.model,
                          choices: [makeChoice({ content: textToEmit })]
                        });
                      }
                      contentEmitBuffer = contentEmitBuffer.substring(flushIndex);
                      break; // wait for more chunks
                    }
                  } else {
                    // Inside tool
                    const endIdx = contentEmitBuffer.indexOf(TOOL_END);
                    if (endIdx !== -1) {
                      let toolJsonStr = contentEmitBuffer.substring(0, endIdx).trim();
                      
                      try {
                        const toolCallObj = robustParseJSON(toolJsonStr);
                        
                        if (!toolCallObj) throw new Error('Empty tool call');

                        // Extract name from XML attribute first, then fall back to JSON
                        const nameMatch = toolJsonStr.match(/<tool_call\s+name="([^"]+)"/);
                        let toolName = nameMatch ? nameMatch[1] : toolCallObj.name || '';

                        // Extract arguments - handle different formats
                        let toolArgs: Record<string, unknown> = {};
                        if (toolCallObj.arguments && typeof toolCallObj.arguments === 'object') {
                          toolArgs = toolCallObj.arguments;
                        } else {
                          // Arguments are the whole object (except name if in JSON)
                          const keys = Object.keys(toolCallObj).filter(k => k !== 'name');
                          for (const k of keys) {
                            toolArgs[k] = toolCallObj[k];
                          }
                        }

                        const toolId = 'call_' + uuidv4();

                        await writeEvent({
                          id: completionId,
                          object: 'chat.completion.chunk',
                          created: Math.floor(Date.now() / 1000),
                          model: body.model,
                          choices: [makeChoice({
                            tool_calls: [{
                              index: emittedToolCallCount,
                              id: toolId,
                              type: 'function',
                              function: {
                                name: toolName,
                                arguments: JSON.stringify(toolArgs)
                              }
                            }]
                          })]
                        });
                        emittedToolCallCount++;
                      } catch (e) {
                        // Failed to parse tool call JSON, emit as regular text
                        
                        if (emittedToolCallCount === 0) {
                          await writeEvent({
                            id: completionId,
                            object: 'chat.completion.chunk',
                            created: Math.floor(Date.now() / 1000),
                            model: body.model,
                            choices: [makeChoice({ content: TOOL_START + toolJsonStr + TOOL_END })]
                          });
                        }
                      }
                      
                      insideTool = false;
                      contentEmitBuffer = contentEmitBuffer.substring(endIdx + TOOL_END.length);
                    } else {
                      // Waiting for TOOL_END, buffer the content
                      break;
                    }
                  }
                }
              }
            }
          } catch (e) {
            // parse error, ignore partial chunk
          }
        }
      }

      // Flush any remaining content emit buffer
      if (!insideTool && contentEmitBuffer.length > 0 && emittedToolCallCount === 0) {
        await writeEvent({
          id: completionId,
          object: 'chat.completion.chunk',
          created: Math.floor(Date.now() / 1000),
          model: body.model,
          choices: [makeChoice({ content: contentEmitBuffer })]
        });
      }
  
      // Send finish reason
      const usage = {
        prompt_tokens: promptTokens,
        completion_tokens: completionTokens,
        total_tokens: promptTokens + completionTokens,
        prompt_tokens_details: {
          cached_tokens: 0 // Mock cache compatibility
        }
      };
  
      const finalFinishReason = emittedToolCallCount > 0 ? 'tool_calls' : 'stop';
  
      await writeEvent({
        id: completionId,
        object: 'chat.completion.chunk',
        created: Math.floor(Date.now() / 1000),
        model: body.model,
        choices: [makeChoice({}, finalFinishReason)],
        usage: usage
      });
      await streamWriter.write('data: [DONE]\n\n');

    });
  } catch (err: any) {
    console.error('Error in chatCompletions:', err);
    const errMessage = err?.message || String(err);

    let status = 500;
    let code = 'upstream_error';
    if (/account is suspended/i.test(errMessage)) {
      status = 403;
      code = 'deepseek_account_suspended';
    } else if (/login is required/i.test(errMessage)) {
      status = 401;
      code = 'deepseek_login_required';
    } else if (/chat input unavailable|Timeout waiting for chat input/i.test(errMessage)) {
      status = 409;
      code = 'deepseek_chat_unavailable';
    }

    return c.json({ error: { message: errMessage, type: code, code } }, status as any);
  }
}
