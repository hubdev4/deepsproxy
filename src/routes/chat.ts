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
import { OpenAIRequest, ChoiceDelta, Message, ToolCall, Usage } from '../utils/types.ts';
import { robustParseJSON } from '../utils/json.ts';
import { shortHash, snippet, trace, TraceContext } from '../utils/trace.ts';

const TOOL_START = '<tool_call>';
const TOOL_END = '</tool_call>';
const TOOL_OPEN_RE = /<tool_call\b[^>]*>/i;
const MAX_TOOL_CALLS_PER_RESPONSE = Number.parseInt(process.env.MAX_TOOL_CALLS_PER_RESPONSE || '1', 10);

type EmitChunk = (data: any) => Promise<void>;

let chatQueue: Promise<void> = Promise.resolve();
let queuedChatRequests = 0;

async function acquireChatLock(ctx?: TraceContext): Promise<() => void> {
  queuedChatRequests++;
  const queueDepthAtEntry = queuedChatRequests;
  trace(ctx, 'queue_enter', { queueDepthAtEntry });
  const previous = chatQueue;
  let releaseCurrent!: () => void;
  chatQueue = new Promise<void>(resolve => {
    releaseCurrent = resolve;
  });
  const waitStart = Date.now();
  await previous.catch(() => {});
  queuedChatRequests--;
  trace(ctx, 'queue_acquired', { waitedMs: Date.now() - waitStart, queuedBehind: queuedChatRequests });
  let released = false;
  return () => {
    if (released) return;
    released = true;
    trace(ctx, 'queue_release', { queuedBehind: queuedChatRequests });
    releaseCurrent();
  };
}

interface ParsedCompletion {
  content: string;
  reasoningContent: string;
  toolCalls: ToolCall[];
  finishReason: string;
  usage: Usage;
}

function messageContentToString(content: any): string {
  if (Array.isArray(content)) {
    return content.map((c: any) => c.text || JSON.stringify(c)).join('\n');
  }
  if (typeof content === 'object' && content !== null) {
    return JSON.stringify(content);
  }
  return content || '';
}

function serializeOpenAIMessages(messages: Message[]) {
  let prompt = '';
  let systemPrompt = '';

  for (const msg of messages) {
    const contentStr = messageContentToString(msg.content);

    if (msg.role === 'system') {
      systemPrompt += contentStr + '\n\n';
      continue;
    }

    if (msg.role === 'user') {
      prompt += `<message role="user">\n${contentStr}\n</message>\n\n`;
      continue;
    }

    if (msg.role === 'assistant') {
      let assistantContent = contentStr;
      if ((msg as any).reasoning_content) {
        assistantContent = `<think>\n${(msg as any).reasoning_content}\n</think>\n${assistantContent}`;
      }
      if (msg.tool_calls && Array.isArray(msg.tool_calls)) {
        for (const tc of msg.tool_calls) {
          let args = tc.function?.arguments || '{}';
          if (typeof args !== 'string') args = JSON.stringify(args);
          assistantContent += `\n<tool_call>{"name": "${tc.function?.name}", "arguments": ${args}}</tool_call>`;
        }
      }
      prompt += `<message role="assistant">\n${assistantContent.trim()}\n</message>\n\n`;
      continue;
    }

    if (msg.role === 'tool' || msg.role === 'function') {
      prompt += `<message role="${msg.role}" name="${msg.name || msg.tool_call_id || 'tool'}">\n${contentStr}\n</message>\n\n`;
      continue;
    }

    prompt += `<message role="${msg.role}">\n${contentStr}\n</message>\n\n`;
  }

  return { prompt, systemPrompt };
}

// DeepSeek web rejects long browser/API prompts before generation with an SSE
// hint error: "Conteúdo muito longo. Encurte e tente novamente."  The previous
// 56k-token estimate allowed ~196k chars and reproduced empty Hermes replies at
// ~164k chars. Keep a conservative request budget so Hermes history/tool output
// is truncated before DeepSeek rejects it.
const MAX_INPUT_TOKENS = 32_000;
const CHARS_PER_TOKEN = 3.5;

function estimateTokens(text: string): number {
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}

function cloneMessageWithContent(msg: Message, content: string): Message {
  return { ...(msg as any), content } as Message;
}

function clipTextToTokenBudget(text: string, maxTokens: number, label: string): string {
  const maxChars = Math.max(0, Math.floor(maxTokens * CHARS_PER_TOKEN));
  if (text.length <= maxChars) return text;
  if (maxChars < 200) return text.slice(0, maxChars);

  const marker = `\n\n[${label} truncated by deepsproxy to fit the browser-backed DeepSeek input budget; middle omitted, start and end preserved.]\n\n`;
  const remaining = Math.max(0, maxChars - marker.length);
  const headChars = Math.ceil(remaining * 0.65);
  const tailChars = remaining - headChars;
  return `${text.slice(0, headChars)}${marker}${tailChars > 0 ? text.slice(-tailChars) : ''}`;
}

function messageBudgetText(msg: Message): string {
  const contentStr = messageContentToString(msg.content);

  if (msg.role === 'assistant') {
    let text = contentStr;
    if ((msg as any).reasoning_content) {
      text += `\n<think>\n${(msg as any).reasoning_content}\n</think>`;
    }
    if (msg.tool_calls && Array.isArray(msg.tool_calls)) {
      for (const tc of msg.tool_calls) {
        let args = tc.function?.arguments || '{}';
        if (typeof args !== 'string') args = JSON.stringify(args);
        text += `\n<tool_call>{"name": "${tc.function?.name}", "arguments": ${args}}</tool_call>`;
      }
    }
    return text;
  }

  if (msg.role === 'tool' || msg.role === 'function') {
    return `${msg.name || msg.tool_call_id || 'tool'}\n${contentStr}`;
  }

  return contentStr;
}

function estimateToolsInstructionTokens(body: OpenAIRequest): number {
  const bodyAny = body as any;
  if (!bodyAny.tools || !Array.isArray(bodyAny.tools) || bodyAny.tools.length === 0) return 0;
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
  // Mirror appendToolInstructions() overhead, conservatively. Tool schemas are
  // injected into the DeepSeek prompt text, so they must count against the real
  // browser/API limit even though they are not message content.
  return estimateTokens(JSON.stringify(formattedTools, null, 2)) + 300;
}

function truncateMessagesToContext(messages: Message[], reservedTokens = 0): Message[] {
  const systemMsgs = messages.filter(m => m.role === 'system');
  const otherMsgs = messages.filter(m => m.role !== 'system');

  let systemTokens = reservedTokens;
  for (const msg of systemMsgs) {
    systemTokens += estimateTokens(messageBudgetText(msg));
  }

  const availableTokens = MAX_INPUT_TOKENS - systemTokens;
  if (availableTokens <= 0) {
    console.warn(`[chat] System/tools prompt (${systemTokens} est. tokens) exceeds budget. Keeping last 2 messages only.`);
    return [...systemMsgs, ...otherMsgs.slice(-2)];
  }

  const keptByIndex = new Map<number, Message>();
  let usedTokens = 0;

  const latestUserIndex = (() => {
    for (let i = otherMsgs.length - 1; i >= 0; i--) {
      if (otherMsgs[i].role === 'user') return i;
    }
    return -1;
  })();

  const keepMessage = (index: number, msg: Message, tokens: number) => {
    keptByIndex.set(index, msg);
    usedTokens += tokens;
  };

  // Agent clients send self-contained histories. The active user objective is
  // usually older than a huge tool result (e.g. skill_view output). If a single
  // trailing tool message exceeds the conservative browser budget, the old
  // suffix-only algorithm kept zero non-system messages and DeepSeek received
  // only the system/tool instructions, which made it greet/reset. Always reserve
  // space for the latest user intent, clipping only if that user message itself
  // is oversized.
  if (latestUserIndex !== -1) {
    const latestUser = otherMsgs[latestUserIndex];
    const userTokens = estimateTokens(messageBudgetText(latestUser)) + 16;
    if (userTokens <= availableTokens) {
      keepMessage(latestUserIndex, latestUser, userTokens);
    } else {
      const clippedTokens = Math.max(1, availableTokens - 16);
      const clipped = cloneMessageWithContent(
        latestUser,
        clipTextToTokenBudget(messageContentToString(latestUser.content), clippedTokens, 'user message')
      );
      keepMessage(latestUserIndex, clipped, estimateTokens(messageBudgetText(clipped)) + 16);
    }
  }

  for (let i = otherMsgs.length - 1; i >= 0; i--) {
    if (keptByIndex.has(i)) continue;
    const msg = otherMsgs[i];
    const msgTokens = estimateTokens(messageBudgetText(msg)) + 16; // overhead per message
    const remainingTokens = availableTokens - usedTokens;
    if (remainingTokens <= 16) break;

    if (msgTokens <= remainingTokens) {
      keepMessage(i, msg, msgTokens);
      continue;
    }

    // Preserve a bounded version of the most recent oversized tool/function
    // result instead of dropping the entire post-tool context. This is the
    // common Hermes failure mode after large skill_view/read_file outputs.
    if ((msg.role === 'tool' || msg.role === 'function') && i === otherMsgs.length - 1) {
      const clipped = cloneMessageWithContent(
        msg,
        clipTextToTokenBudget(messageContentToString(msg.content), Math.max(1, remainingTokens - 16), 'tool response')
      );
      keepMessage(i, clipped, estimateTokens(messageBudgetText(clipped)) + 16);
    }
    break;
  }

  const truncated = Array.from(keptByIndex.entries())
    .sort(([a], [b]) => a - b)
    .map(([, msg]) => msg);
  if (truncated.length < otherMsgs.length) {
    console.log(`[chat] Truncated messages: ${otherMsgs.length} -> ${truncated.length} (removed ${otherMsgs.length - truncated.length} non-system messages, ~${usedTokens} est. input tokens, reserved ~${reservedTokens}, latestUserPreserved=${latestUserIndex !== -1 && keptByIndex.has(latestUserIndex)})`);
  }

  return [...systemMsgs, ...truncated];
}

function appendToolInstructions(systemPrompt: string, body: OpenAIRequest): string {
  const bodyAny = body as any;
  if (!bodyAny.tools || !Array.isArray(bodyAny.tools) || bodyAny.tools.length === 0) {
    return systemPrompt;
  }

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

  systemPrompt += `\n\n# CONVERSATION HISTORY FORMAT\nPrior OpenAI messages are serialized below as <message role="..."> blocks for context only. Do not copy, quote, or continue the history markup. In your response, act only as the next assistant turn. Never emit transcript labels such as "User:", "Assistant:", or "Tool Response (...)" as assistant content.\n\n# TOOLS AVAILABLE\nYou have access to the following tools:\n${toolsJson}\n\nTo use a tool, you MUST output a JSON object wrapped EXACTLY in these tags:\n<tool_call>\n{"name": "tool_name", "arguments": {"param_name": "value"}}\n</tool_call>\n\nRULES:\n1. Call at most one tool in this response unless the user explicitly asked for parallel execution. Prefer the single highest-signal diagnostic/action instead of shotgun probing alternatives.\n2. Do NOT output any other text before or after your <tool_call> block. Wait for the user to provide the tool response.\n3. The JSON must be valid and accurately follow the tool's parameters.\n\n`;

  if (bodyAny.tool_choice && typeof bodyAny.tool_choice === 'object' && bodyAny.tool_choice.function) {
    const forcedTool = bodyAny.tool_choice.function.name;
    systemPrompt += `CRITICAL: You MUST call the tool "${forcedTool}" in this response.\n\n`;
  }

  return systemPrompt;
}

function decodeXmlEntities(value: string): string {
  return value
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&');
}

function coerceParameterValue(rawValue: string): unknown {
  const value = decodeXmlEntities(rawValue.trim());
  if (value === 'true') return true;
  if (value === 'false') return false;
  if (value === 'null') return null;
  if (/^-?\d+(?:\.\d+)?$/.test(value)) return Number(value);
  if ((value.startsWith('{') && value.endsWith('}')) || (value.startsWith('[') && value.endsWith(']'))) {
    try { return JSON.parse(value); } catch {}
  }
  return value;
}

function extractToolName(openTag: string, block: string): string {
  const combined = `${openTag}\n${block}`;
  const attrMatch = combined.match(/<tool_call\b[^>]*\bname\s*=\s*["']([^"']+)["']/i);
  if (attrMatch) return attrMatch[1];

  const nameTagMatch = block.match(/<name>([\s\S]*?)<\/name>/i);
  if (nameTagMatch) return decodeXmlEntities(nameTagMatch[1].trim());

  return '';
}

function inferToolNameFromParameters(args: Record<string, unknown>, tools: any[]): string {
  const argKeys = Object.keys(args);
  if (argKeys.length === 0 || !Array.isArray(tools)) return '';

  const matches = tools.filter((tool: any) => {
    const fn = tool?.type === 'function' ? tool.function : tool?.function;
    const properties = fn?.parameters?.properties || {};
    return argKeys.every(k => Object.prototype.hasOwnProperty.call(properties, k));
  });

  if (matches.length === 1) {
    const fn = matches[0]?.type === 'function' ? matches[0].function : matches[0]?.function;
    return fn?.name || '';
  }

  return '';
}

function parseXmlParameterToolCall(block: string, openTag: string, tools: any[]): any | null {
  const args: Record<string, unknown> = {};
  const parameterRe = /<parameter\b[^>]*\bname\s*=\s*["']([^"']+)["'][^>]*>([\s\S]*?)<\/parameter>/gi;
  let match: RegExpExecArray | null;
  while ((match = parameterRe.exec(block)) !== null) {
    args[match[1]] = coerceParameterValue(match[2]);
  }

  if (Object.keys(args).length === 0) return null;

  const toolName = extractToolName(openTag, block) || inferToolNameFromParameters(args, tools);
  if (!toolName) return null;

  return { name: toolName, arguments: args };
}

function parseToolCallBlock(block: string, openTag: string, tools: any[]): any {
  const parsedXml = parseXmlParameterToolCall(block, openTag, tools);
  if (parsedXml) return parsedXml;

  const parsedJson = robustParseJSON(block);
  if (!parsedJson) throw new Error('Empty tool call');

  const attrToolName = extractToolName(openTag, block);
  if (attrToolName && !parsedJson.name) parsedJson.name = attrToolName;

  if (!parsedJson.name && parsedJson && typeof parsedJson === 'object' && !Array.isArray(parsedJson)) {
    const knownToolNames = new Set((Array.isArray(tools) ? tools : [])
      .map((tool: any) => (tool?.type === 'function' ? tool.function?.name : tool?.function?.name || tool?.name))
      .filter((name: any) => typeof name === 'string' && name.length > 0));
    const shorthandKeys = Object.keys(parsedJson).filter(key => knownToolNames.has(key));

    if (shorthandKeys.length === 1) {
      const toolName = shorthandKeys[0];
      const shorthandArgs = parsedJson[toolName];
      if (shorthandArgs && typeof shorthandArgs === 'object' && !Array.isArray(shorthandArgs)) {
        return { name: toolName, arguments: shorthandArgs };
      }
    }
  }

  return parsedJson;
}

function findToolOpen(buffer: string): { startIdx: number; endIdx: number; openTag: string } | null {
  const match = buffer.match(TOOL_OPEN_RE);
  if (!match || match.index === undefined) return null;
  return {
    startIdx: match.index,
    endIdx: match.index + match[0].length,
    openTag: match[0]
  };
}

function findPartialToolOpenIndex(buffer: string): number {
  const lower = buffer.toLowerCase();
  const idx = lower.lastIndexOf('<tool_call');
  if (idx !== -1 && lower.indexOf('>', idx) === -1) return idx;

  for (let i = 1; i < TOOL_START.length; i++) {
    if (lower.endsWith(TOOL_START.substring(0, i))) return buffer.length - i;
  }
  return -1;
}

function makeChoice(delta: any, finishReason: string | null = null) {
  return {
    index: 0,
    delta,
    logprobs: null,
    finish_reason: finishReason
  };
}

function makeChunk(completionId: string, model: string, delta: any, finishReason: string | null = null, usage?: Usage) {
  const chunk: any = {
    id: completionId,
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [makeChoice(delta, finishReason)]
  };
  if (usage) chunk.usage = usage;
  return chunk;
}

async function parseDeepSeekStreamToOpenAI(
  deepSeekStream: ReadableStream,
  completionId: string,
  model: string,
  promptTokens: number,
  uiSessionId: string,
  tools: any[] = [],
  emit?: EmitChunk,
  ctx?: TraceContext
): Promise<ParsedCompletion> {
  const reader = deepSeekStream.getReader();
  const decoder = new TextDecoder();

  let currentAppendPath = '';
  let currentFragmentType = '';
  let reasoningContent = '';
  let content = '';
  let contentEmitBuffer = '';
  let insideTool = false;
  let currentToolOpenTag = TOOL_START;
  let emittedToolCallCount = 0;
  let completionTokens = 0;
  const toolCalls: ToolCall[] = [];
  let buffer = '';
  let pendingToolLeadIn = '';
  let rawChunkCount = 0;
  let dataLineCount = 0;
  let parsedJsonLineCount = 0;
  let foundTextChunkCount = 0;
  let emittedContentChunkCount = 0;
  let ignoredNoTextCount = 0;
  let parseErrorCount = 0;
  let providerErrorContent = '';
  let providerErrorFinishReason = '';
  let firstTextAtMs: number | null = null;
  const parseStart = Date.now();
  const deferContentUntilToolDecision = emit !== undefined && tools.length > 0;
  let deferredContentFlushed = false;
  let suppressedToolLeadInLength = 0;

  trace(ctx, 'parse_start', { stream: Boolean(emit), model, promptTokens, toolsCount: tools.length, uiSessionIdHash: shortHash(uiSessionId) });

  const emitContent = async (text: string) => {
    if (!text || emittedToolCallCount > 0) return;
    if (firstTextAtMs === null) firstTextAtMs = Date.now() - parseStart;
    content += text;

    if (deferContentUntilToolDecision) {
      trace(ctx, 'defer_content_until_tool_decision', { length: text.length, totalContentLength: content.length, snippet: snippet(text, 240) });
      return;
    }

    emittedContentChunkCount++;
    trace(ctx, 'emit_content', { length: text.length, totalContentLength: content.length, snippet: snippet(text, 240) });
    if (emit) await emit(makeChunk(completionId, model, { content: text }));
  };

  const flushDeferredContent = async () => {
    if (!emit || !deferContentUntilToolDecision || deferredContentFlushed || emittedToolCallCount > 0 || !content) return;
    deferredContentFlushed = true;
    emittedContentChunkCount++;
    trace(ctx, 'flush_deferred_content', { length: content.length, snippet: snippet(content, 240) });
    await emit(makeChunk(completionId, model, { content }));
  };

  const parseRecoverableToolCallBlock = (block: string, openTag: string): any => {
    try {
      return parseToolCallBlock(block, openTag, tools);
    } catch {}

    const args: Record<string, unknown> = {};
    const closedParameterRe = /<parameter\b[^>]*\bname\s*=\s*["']([^"']+)["'][^>]*>([\s\S]*?)<\/parameter>/gi;
    let match: RegExpExecArray | null;
    let lastClosedEnd = 0;
    while ((match = closedParameterRe.exec(block)) !== null) {
      args[match[1]] = coerceParameterValue(match[2]);
      lastClosedEnd = closedParameterRe.lastIndex;
    }

    const tail = block.substring(lastClosedEnd);
    const unclosedParameterMatch = tail.match(/<parameter\b[^>]*\bname\s*=\s*["']([^"']+)["'][^>]*>([\s\S]*)$/i);
    if (unclosedParameterMatch) {
      args[unclosedParameterMatch[1]] = coerceParameterValue(unclosedParameterMatch[2]);
    }

    if (Object.keys(args).length === 0) throw new Error('Unrecoverable tool call');
    const toolName = extractToolName(openTag, block) || inferToolNameFromParameters(args, tools);
    if (!toolName) throw new Error('Recoverable tool call missing name');
    return { name: toolName, arguments: args };
  };

  const emitToolCallFromBlock = async (toolBlock: string, openTag: string) => {
    const effectiveMaxToolCalls = Number.isFinite(MAX_TOOL_CALLS_PER_RESPONSE) && MAX_TOOL_CALLS_PER_RESPONSE > 0
      ? MAX_TOOL_CALLS_PER_RESPONSE
      : 1;
    if (emittedToolCallCount >= effectiveMaxToolCalls) {
      trace(ctx, 'tool_call_dropped_over_cap', {
        emittedToolCallCount,
        maxToolCalls: effectiveMaxToolCalls,
        toolBlockLength: toolBlock.length,
        toolBlockSnippet: snippet(toolBlock, 300)
      });
      return;
    }

    const toolCallObj = parseRecoverableToolCallBlock(toolBlock, openTag);
    const toolName = toolCallObj.name || '';

    let toolArgs: Record<string, unknown> = {};
    if (toolCallObj.arguments && typeof toolCallObj.arguments === 'object') {
      toolArgs = toolCallObj.arguments;
    } else {
      const keys = Object.keys(toolCallObj).filter(k => k !== 'name');
      for (const k of keys) toolArgs[k] = toolCallObj[k];
    }

    if (!toolName) throw new Error('Tool call missing name');

    if (emittedToolCallCount === 0) {
      suppressedToolLeadInLength += content.length + pendingToolLeadIn.length;
      if (content || pendingToolLeadIn) {
        trace(ctx, 'suppress_tool_lead_in', {
          contentLength: content.length,
          pendingLeadInLength: pendingToolLeadIn.length,
          snippet: snippet(`${content}${pendingToolLeadIn}`, 300)
        });
      }
      content = '';
      pendingToolLeadIn = '';
    }

    const toolId = 'call_' + uuidv4();
    const toolCall: ToolCall = {
      index: emittedToolCallCount,
      id: toolId,
      type: 'function',
      function: { name: toolName, arguments: JSON.stringify(toolArgs) }
    };
    toolCalls.push(toolCall);
    trace(ctx, 'emit_tool_call', { toolIndex: emittedToolCallCount, toolName, argsHash: shortHash(toolArgs), argsLength: JSON.stringify(toolArgs).length });
    if (emit) await emit(makeChunk(completionId, model, { tool_calls: [toolCall] }));
    emittedToolCallCount++;
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    rawChunkCount++;
    const decoded = decoder.decode(value, { stream: true });
    if (rawChunkCount <= 5 || rawChunkCount % 50 === 0) {
      trace(ctx, 'deepseek_raw_chunk', { rawChunkCount, bytes: value?.byteLength || 0, decodedLength: decoded.length, snippet: snippet(decoded, 300) });
    }
    buffer += decoded;
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.startsWith('data: ')) continue;

      const dataStr = trimmed.slice(6);
      dataLineCount++;
      if (dataStr === '[DONE]') {
        trace(ctx, 'deepseek_done_line', { dataLineCount });
        continue;
      }

      try {
        const chunk = JSON.parse(dataStr);
        parsedJsonLineCount++;
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

        if (dsMessageId) updateSessionParent(uiSessionId, dsMessageId);

        let vStr = '';
        let foundStr = false;
        let isThinkingChunk = false;

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
          if (lastFrag && lastFrag.type) currentFragmentType = lastFrag.type;
        }

        if (currentAppendPath.includes('thinking_content') ||
            currentAppendPath.includes('THINK') ||
            (currentAppendPath.includes('fragments/-1/content') && currentFragmentType === 'THINK')) {
          isThinkingChunk = true;
        }

        if (chunk?.type === 'error' && typeof chunk.content === 'string') {
          providerErrorContent = chunk.content;
          providerErrorFinishReason = typeof chunk.finish_reason === 'string' ? chunk.finish_reason : 'provider_error';
          trace(ctx, 'deepseek_provider_error_hint', {
            content: providerErrorContent,
            finishReason: providerErrorFinishReason,
            clearResponse: chunk.clear_response === true
          });
          if (!content && toolCalls.length === 0 && !reasoningContent) {
            await emitContent(`[DeepSeek erro: ${providerErrorContent}]`);
          }
        }

        if (!foundStr || vStr === '' || vStr === 'FINISHED') {
          ignoredNoTextCount++;
          if (ignoredNoTextCount <= 10 || ignoredNoTextCount % 50 === 0) {
            trace(ctx, 'deepseek_no_text_chunk', {
              ignoredNoTextCount,
              path: currentAppendPath,
              fragmentType: currentFragmentType,
              hasV: chunk.v !== undefined,
              vType: typeof chunk.v,
              p: chunk.p,
              keys: Object.keys(chunk).slice(0, 12)
            });
          }
          continue;
        }

        foundTextChunkCount++;
        if (foundTextChunkCount <= 20 || foundTextChunkCount % 50 === 0) {
          trace(ctx, isThinkingChunk ? 'deepseek_thinking_text' : 'deepseek_content_text', {
            foundTextChunkCount,
            length: vStr.length,
            path: currentAppendPath,
            fragmentType: currentFragmentType,
            snippet: snippet(vStr, 240)
          });
        }

        if (isThinkingChunk) {
          reasoningContent += vStr;
          const delta: ChoiceDelta = { reasoning_content: vStr };
          if (emit) await emit(makeChunk(completionId, model, delta));
          continue;
        }

        contentEmitBuffer += vStr;

        while (contentEmitBuffer.length > 0) {
          if (!insideTool) {
            const toolOpen = findToolOpen(contentEmitBuffer);
            if (toolOpen) {
              // Once a tool call appears, do not emit the lead-in text as
              // assistant content. OpenAI-compatible clients expect the whole
              // assistant turn to be a structured tool_calls message.
              pendingToolLeadIn += contentEmitBuffer.substring(0, toolOpen.startIdx);
              insideTool = true;
              currentToolOpenTag = toolOpen.openTag;
              contentEmitBuffer = contentEmitBuffer.substring(toolOpen.endIdx);
              continue;
            }

            const partialStartIdx = findPartialToolOpenIndex(contentEmitBuffer);
            const flushIndex = partialStartIdx === -1 ? contentEmitBuffer.length : partialStartIdx;

            const textToEmit = contentEmitBuffer.substring(0, flushIndex);
            await emitContent(textToEmit);
            contentEmitBuffer = contentEmitBuffer.substring(flushIndex);
            break;
          }

          const lowerBuffer = contentEmitBuffer.toLowerCase();
          const endIdx = lowerBuffer.indexOf(TOOL_END);
          if (endIdx === -1) break;

          const toolBlock = contentEmitBuffer.substring(0, endIdx).trim();
          try {
            await emitToolCallFromBlock(toolBlock, currentToolOpenTag);
            pendingToolLeadIn = '';
          } catch (e: any) {
            // Never leak internal tool-call XML to the user-visible content.
            // If the call cannot be parsed, restore any normal text that came
            // before it so the OpenAI response is not silently empty.
            console.warn('[chat] Dropping malformed tool call block:', e);
            trace(ctx, 'tool_call_parse_failed', {
              message: e?.message || String(e),
              toolBlockLength: toolBlock.length,
              toolBlockSnippet: snippet(toolBlock, 500),
              pendingLeadInLength: pendingToolLeadIn.length,
              emittedToolCallCount
            });
            if (emittedToolCallCount === 0 && pendingToolLeadIn.trim().length > 0) {
              await emitContent(pendingToolLeadIn);
            }
            pendingToolLeadIn = '';
          }

          insideTool = false;
          currentToolOpenTag = TOOL_START;
          contentEmitBuffer = contentEmitBuffer.substring(endIdx + TOOL_END.length);
        }
      } catch (e: any) {
        parseErrorCount++;
        trace(ctx, 'deepseek_parse_error', { parseErrorCount, message: e?.message || String(e), dataSnippet: snippet(dataStr, 300) });
        // Ignore partial or malformed DeepSeek chunks.
      }
    }
  }

  if (insideTool && contentEmitBuffer.trim().length > 0) {
    try {
      await emitToolCallFromBlock(contentEmitBuffer.trim(), currentToolOpenTag);
      pendingToolLeadIn = '';
    } catch (e: any) {
      console.warn('[chat] Dropping unclosed malformed tool call at end of stream:', e);
      trace(ctx, 'tool_call_unclosed_failed', {
        message: e?.message || String(e),
        bufferLength: contentEmitBuffer.length,
        bufferSnippet: snippet(contentEmitBuffer, 500),
        pendingLeadInLength: pendingToolLeadIn.length,
        emittedToolCallCount
      });
      if (emittedToolCallCount === 0 && pendingToolLeadIn.trim().length > 0) {
        await emitContent(pendingToolLeadIn);
      }
      pendingToolLeadIn = '';
    }
  }

  if (!insideTool && contentEmitBuffer.length > 0 && emittedToolCallCount === 0) {
    await emitContent(contentEmitBuffer);
  }

  await flushDeferredContent();

  const usage: Usage = {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: promptTokens + completionTokens,
    prompt_tokens_details: { cached_tokens: 0 }
  };

  trace(ctx, 'parse_complete', {
    elapsedMs: Date.now() - parseStart,
    rawChunkCount,
    dataLineCount,
    parsedJsonLineCount,
    foundTextChunkCount,
    ignoredNoTextCount,
    parseErrorCount,
    emittedContentChunkCount,
    contentLength: content.length,
    reasoningLength: reasoningContent.length,
    toolCallsCount: toolCalls.length,
    finishReason: emittedToolCallCount > 0 ? 'tool_calls' : (providerErrorFinishReason || 'stop'),
    providerErrorContent: providerErrorContent ? snippet(providerErrorContent, 300) : '',
    firstTextAtMs,
    leftoverBufferLength: buffer.length,
    contentEmitBufferLength: contentEmitBuffer.length,
    insideTool,
    suppressedToolLeadInLength
  });

  return {
    content,
    reasoningContent,
    toolCalls,
    finishReason: emittedToolCallCount > 0 ? 'tool_calls' : (providerErrorFinishReason || 'stop'),
    usage
  };
}

export async function chatCompletions(c: Context) {
  const requestId = c.req.header('x-request-id') || c.req.header('x-hermes-session-id') || uuidv4();
  let ctx: TraceContext = { requestId };
  const requestStart = Date.now();
  try {
    const body: OpenAIRequest = await c.req.json();
    const isStream = body.stream ?? false;
    const rawMessages = body.messages || [];
    const reservedTokens = estimateToolsInstructionTokens(body);
    const messages = truncateMessagesToContext(rawMessages, reservedTokens);

    const serialized = serializeOpenAIMessages(messages);
    const systemPrompt = appendToolInstructions(serialized.systemPrompt, body);
    const finalPrompt = systemPrompt ? `${systemPrompt}\n${serialized.prompt}` : serialized.prompt;

    const isThinkingModel = body.model.includes('thinking');
    const isProModel = body.model.includes('pro');
    const completionId = 'chatcmpl-' + uuidv4();
    ctx = { requestId, completionId };
    const promptTokens = Math.ceil(finalPrompt.length / 3.5);

    trace(ctx, 'request_start', {
      model: body.model,
      stream: isStream,
      messagesCount: rawMessages.length,
      truncatedMessagesCount: messages.length,
      toolsCount: Array.isArray((body as any).tools) ? (body as any).tools.length : 0,
      toolChoice: (body as any).tool_choice ? snippet(JSON.stringify((body as any).tool_choice), 300) : null,
      promptLength: finalPrompt.length,
      promptTokens,
      promptHash: shortHash(finalPrompt),
      messageSummary: rawMessages.map((m: any, index: number) => ({
        index,
        role: m.role,
        contentLength: messageContentToString(m.content).length,
        hasToolCalls: Array.isArray(m.tool_calls),
        toolCallsCount: Array.isArray(m.tool_calls) ? m.tool_calls.length : 0,
        toolCallId: m.tool_call_id || null,
        name: m.name || null,
        contentSnippet: snippet(messageContentToString(m.content), 180)
      })).slice(-20),
      truncatedMessageSummary: messages.map((m: any, index: number) => ({
        index,
        role: m.role,
        contentLength: messageContentToString(m.content).length,
        hasToolCalls: Array.isArray(m.tool_calls),
        toolCallsCount: Array.isArray(m.tool_calls) ? m.tool_calls.length : 0,
        toolCallId: m.tool_call_id || null,
        name: m.name || null,
        contentSnippet: snippet(messageContentToString(m.content), 180)
      })).slice(-20)
    });

    // DeepSeek web-chat access is browser-backed and uses one persistent page/profile.
    // Concurrent OpenAI requests race on the same page route/textarea and can leave
    // the provider returning empty messages or "Timeout waiting for chat input".
    // Serialize complete chat turns, including streamed turns, to keep the browser
    // state and PoW-header capture deterministic.
    const releaseChatLock = await acquireChatLock(ctx);
    if (queuedChatRequests > 0) {
      console.log(`[chat] Serialized DeepSeek request; ${queuedChatRequests} request(s) still queued`);
    }

    let deepSeekStream: ReadableStream;
    let uiSessionId = '';
    let retries = 3;
    let streamResponseReturned = false;
    try {
      while (retries > 0) {
        try {
          // OpenAI chat/completions requests are self-contained: the caller sends
          // the full message history every time. Always start a fresh DeepSeek
          // browser turn so DeepSeek's stateful parent_message_id cannot drift
          // from compressed/edited OpenAI histories and produce empty replies.
          trace(ctx, 'deepseek_create_attempt', { attempt: 4 - retries, retriesRemainingBeforeAttempt: retries - 1, isThinkingModel, isProModel });
          const result = await createDeepSeekStream(finalPrompt, isThinkingModel, isProModel, null, ctx);
          deepSeekStream = result.stream;
          uiSessionId = result.uiSessionId;
          trace(ctx, 'deepseek_stream_created', { attempt: 4 - retries, uiSessionIdHash: shortHash(uiSessionId) });
          break;
        } catch (err: any) {
          const errMessage = err?.message || String(err);
          trace(ctx, 'deepseek_create_error', { attempt: 4 - retries, retriesRemainingAfterError: retries - 1, message: errMessage, stack: err?.stack });
          const nonRetryableBrowserState = /account is suspended|login is required|chat input unavailable|Timeout waiting for chat input/i.test(errMessage);
          retries--;
          if (nonRetryableBrowserState || retries === 0) throw err;
          await new Promise(r => setTimeout(r, 1000));
        }
      }

      if (!isStream) {
        const parsed = await parseDeepSeekStreamToOpenAI(
          deepSeekStream!,
          completionId,
          body.model,
          promptTokens,
          uiSessionId,
          (body as any).tools || [],
          undefined,
          ctx
        );

        const message: any = {
          role: 'assistant',
          content: parsed.toolCalls.length > 0 ? null : parsed.content
        };
        if (parsed.reasoningContent) message.reasoning_content = parsed.reasoningContent;
        if (parsed.toolCalls.length > 0) message.tool_calls = parsed.toolCalls;

        trace(ctx, 'response_nonstream_ready', {
          elapsedMs: Date.now() - requestStart,
          status: 200,
          contentLength: parsed.content.length,
          reasoningLength: parsed.reasoningContent.length,
          toolCallsCount: parsed.toolCalls.length,
          finishReason: parsed.finishReason,
          contentSnippet: snippet(parsed.content, 300)
        });

        return c.json({
          id: completionId,
          object: 'chat.completion',
          created: Math.floor(Date.now() / 1000),
          model: body.model,
          choices: [{
            index: 0,
            message,
            logprobs: null,
            finish_reason: parsed.finishReason
          }],
          usage: parsed.usage
        });
      }

      c.header('Content-Type', 'text/event-stream');
      c.header('Cache-Control', 'no-cache');
      c.header('Connection', 'keep-alive');

      streamResponseReturned = true;
      return honoStream(c, async (streamWriter: any) => {
        try {
          trace(ctx, 'response_stream_start', { elapsedMs: Date.now() - requestStart });
          const writeEvent = async (data: any) => {
            await streamWriter.write(`data: ${JSON.stringify(data)}\n\n`);
          };

          await writeEvent(makeChunk(completionId, body.model, { role: 'assistant', content: '' }));

          const parsed = await parseDeepSeekStreamToOpenAI(
            deepSeekStream!,
            completionId,
            body.model,
            promptTokens,
            uiSessionId,
            (body as any).tools || [],
            writeEvent,
            ctx
          );

          trace(ctx, 'response_stream_complete', {
            elapsedMs: Date.now() - requestStart,
            contentLength: parsed.content.length,
            reasoningLength: parsed.reasoningContent.length,
            toolCallsCount: parsed.toolCalls.length,
            finishReason: parsed.finishReason,
            contentSnippet: snippet(parsed.content, 300)
          });

          await writeEvent(makeChunk(completionId, body.model, {}, parsed.finishReason, parsed.usage));
          await streamWriter.write('data: [DONE]\n\n');
        } finally {
          releaseChatLock();
        }
      });
    } finally {
      if (!isStream || !streamResponseReturned) releaseChatLock();
    }
  } catch (err: any) {
    console.error('Error in chatCompletions:', err);
    const errMessage = err?.message || String(err);
    trace(ctx, 'request_error_response', { elapsedMs: Date.now() - requestStart, message: errMessage, stack: err?.stack });

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
