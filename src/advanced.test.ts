import test from 'node:test';
import assert from 'node:assert';

process.env.TEST_MOCK_PLAYWRIGHT = 'true';

import { app } from './index.ts';

// Helper to mock the fetch global for testing empty response retry and caching logic
function setupFetchMock(handler: (url: string, init?: RequestInit) => Response | Promise<Response>) {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const urlStr = typeof input === 'string' ? input : ('url' in input ? input.url : String(input));
    if (urlStr.includes('chat.deepseek.com')) {
      return handler(urlStr, init);
    }
    return originalFetch(input, init);
  };
  return () => { globalThis.fetch = originalFetch; };
}

test('multiturn-thinking-tools: serializes complete OpenAI message history', async () => {
  let capturedPrompt = '';

  const restore = setupFetchMock((url, init) => {
    const bodyObj = JSON.parse(init?.body as string || '{}');
    capturedPrompt = bodyObj.prompt;
    const stream = new ReadableStream({
      start(c) {
        c.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
        c.close();
      }
    });
    return new Response(stream, { status: 200 });
  });

  try {
    const req = new Request('http://localhost/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'deepseek-v4-flash-thinking',
        messages: [
          { role: 'user', content: 'hello' },
          { role: 'assistant', content: 'doing something', reasoning_content: 'thinking about hello', tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'test', arguments: '{}' } }] },
          { role: 'tool', name: 'test', content: 'success' }
        ]
      })
    });
    
    const res = await app.fetch(req);
    assert.strictEqual(res.status, 200);

    // Validate that the complete OpenAI history is sent to DeepSeek. Agents
    // need the original user request, assistant tool call, and tool result to
    // produce the post-tool final answer.
    assert.ok(capturedPrompt.includes('<message role="user">\nhello\n</message>'), 'Must include original user message');
    assert.ok(capturedPrompt.includes('<message role="assistant">'), 'Must include assistant history');
    assert.ok(capturedPrompt.includes('<think>\nthinking about hello\n</think>'), 'Must include previous thinking');
    assert.ok(capturedPrompt.includes('<tool_call>{"name": "test", "arguments": {}}</tool_call>'), 'Must include previous tool call');
    assert.ok(capturedPrompt.includes('<message role="tool" name="test">\nsuccess\n</message>'), 'Must include tool response as role-delimited history');
    assert.ok(!capturedPrompt.includes('Tool Response (test):'), 'Prompt should avoid transcript labels that models copy');
  } finally {
    restore();
  }
});

test('post-tool truncation preserves latest user intent and clips oversized tool result', async () => {
  let capturedPrompt = '';
  const hugeToolOutput = `SKILL_VIEW_START\n${'A'.repeat(150_000)}\nSKILL_VIEW_END`;

  const restore = setupFetchMock((url, init) => {
    const bodyObj = JSON.parse(init?.body as string || '{}');
    capturedPrompt = bodyObj.prompt;
    const stream = new ReadableStream({
      start(c) {
        c.enqueue(new TextEncoder().encode('data: {"p":"response/content","v":"continuando"}\n\n'));
        c.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
        c.close();
      }
    });
    return new Response(stream, { status: 200 });
  });

  try {
    const req = new Request('http://localhost/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'deepseek-v4-flash',
        messages: [
          { role: 'system', content: 'Você é Hermes.' },
          { role: 'user', content: 'Pesquise novidades do Hermes usando Chrome via ADB no Samsung A55.' },
          { role: 'assistant', content: null, tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'skill_view', arguments: '{"name":"hermes-agent"}' } }] },
          { role: 'tool', tool_call_id: 'call_1', name: 'skill_view', content: hugeToolOutput }
        ],
        stream: false
      })
    });

    const res = await app.fetch(req);
    assert.strictEqual(res.status, 200);
    await res.json();

    assert.ok(capturedPrompt.includes('<message role="user">\nPesquise novidades do Hermes usando Chrome via ADB no Samsung A55.\n</message>'), 'latest user objective must survive truncation');
    assert.ok(capturedPrompt.includes('<message role="tool" name="skill_view">\nSKILL_VIEW_START'), 'oversized tool result should be clipped, not dropped entirely');
    assert.ok(capturedPrompt.includes('tool response truncated by deepsproxy'), 'clipped tool result should be marked explicitly');
    assert.ok(capturedPrompt.includes('SKILL_VIEW_END'), 'tail of clipped tool result should be preserved');
    assert.ok(capturedPrompt.length < 130_000, 'prompt should stay below the conservative browser-backed budget');
  } finally {
    restore();
  }
});

test('streaming-whitespace: preserves exact whitespace', async () => {
  const restore = setupFetchMock((url) => {
    const stream = new ReadableStream({
      start(c) {
        c.enqueue(new TextEncoder().encode('data: {"v":{"response":{"message_id":1}}}\n\n'));
        c.enqueue(new TextEncoder().encode('data: {"p":"response/content","v":"   "}\n\n'));
        c.enqueue(new TextEncoder().encode('data: {"o":"APPEND","v":"  hello  "}\n\n'));
        c.enqueue(new TextEncoder().encode('data: {"o":"APPEND","v":"\\n\\n  "}\n\n'));
        c.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
        c.close();
      }
    });
    return new Response(stream, { status: 200 });
  });

  try {
    const req = new Request('http://localhost/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'deepseek-v4-flash-thinking', messages: [{role: 'user', content: 'test'}], stream: true })
    });
    
    const res = await app.fetch(req);
    const reader = res.body?.getReader();
    const decoder = new TextDecoder();
    let full = '';
    while (true) {
      const { done, value } = await reader!.read();
      if (done) break;
      const chunk = decoder.decode(value);
      for (const line of chunk.split('\n')) {
        if (line.startsWith('data: ') && line !== 'data: [DONE]') {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.choices?.[0]?.delta?.content) {
              full += data.choices[0].delta.content;
            }
          } catch(e) {}
        }
      }
    }
    
    // We expect exactly: "     hello  \n\n  "
    assert.strictEqual(full, "     hello  \n\n  ");
  } finally {
    restore();
  }
});

test('caching-streaming and cache-control: returns prompt_tokens_details', async () => {
  const restore = setupFetchMock((url) => {
    const stream = new ReadableStream({
      start(c) {
        c.enqueue(new TextEncoder().encode('data: {"v":{"response":{"message_id":1}}}\n\n'));
        c.enqueue(new TextEncoder().encode('data: {"p":"response/content","v":"done"}\n\n'));
        c.enqueue(new TextEncoder().encode('data: {"p":"response/accumulated_token_usage","o":"SET","v":10}\n\n'));
        c.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
        c.close();
      }
    });
    return new Response(stream, { status: 200 });
  });

  try {
    const req = new Request('http://localhost/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'deepseek-v4-flash-thinking', messages: [{role: 'user', content: 'test'}], stream: true })
    });
    
    const res = await app.fetch(req);
    const reader = res.body?.getReader();
    const decoder = new TextDecoder();
    let usageBlock = null;
    while (true) {
      const { done, value } = await reader!.read();
      if (done) break;
      const chunk = decoder.decode(value);
      for (const line of chunk.split('\n')) {
        if (line.startsWith('data: ') && line !== 'data: [DONE]') {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.usage) {
              usageBlock = data.usage;
            }
          } catch(e) {}
        }
      }
    }
    
    assert.ok(usageBlock);
    assert.strictEqual(usageBlock.completion_tokens, 10);
    assert.ok(usageBlock.prompt_tokens > 0);
    assert.strictEqual(usageBlock.prompt_tokens_details.cached_tokens, 0); // Tests caching-streaming shape!
  } finally {
    restore();
  }
});

test('openai-requests-are-stateless: each request starts a fresh DeepSeek turn', async () => {
  let capturedPayloads: any[] = [];

  const restore = setupFetchMock((url, init) => {
    const bodyObj = JSON.parse(init?.body as string || '{}');
    capturedPayloads.push(bodyObj);
    
    // Simulate DeepSeek returning a message_id
    const mockMessageId = capturedPayloads.length === 1 ? 1001 : 1002;
    
    const stream = new ReadableStream({
      start(c) {
        c.enqueue(new TextEncoder().encode(`data: {"v":{"response":{"message_id":${mockMessageId}}}}\n\n`));
        c.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
        c.close();
      }
    });
    return new Response(stream, { status: 200 });
  });

  try {
    process.env.TEST_SESSION_ID = 'test-session-parent-tracking';
    // Turn 1
    const req1 = new Request('http://localhost/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'deepseek-v4-flash-thinking',
        messages: [{ role: 'user', content: 'Turn 1' }]
      })
    });
    
    const res1 = await app.fetch(req1);
    assert.strictEqual(res1.status, 200);
    // Consume the stream to ensure the message_id is processed
    await res1.text();

    // Turn 2
    const req2 = new Request('http://localhost/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'deepseek-v4-flash-thinking',
        messages: [
          { role: 'user', content: 'Turn 1' },
          { role: 'assistant', content: 'Response 1' },
          { role: 'user', content: 'Turn 2' }
        ]
      })
    });
    
    const res2 = await app.fetch(req2);
    assert.strictEqual(res2.status, 200);
    await res2.text();

    assert.strictEqual(capturedPayloads.length, 2);
    // In Turn 1, parent_message_id should be null (mock-session is fresh)
    assert.strictEqual(capturedPayloads[0].parent_message_id, null);
    // OpenAI chat/completions requests are self-contained; the proxy must not
    // reuse DeepSeek's previous parent_message_id because compressed or edited
    // OpenAI histories no longer match the browser-side DeepSeek thread.
    assert.strictEqual(capturedPayloads[1].parent_message_id, null, 'Turn 2 should start a fresh DeepSeek turn');
    assert.strictEqual(
      capturedPayloads[1].prompt,
      '<message role="user">\nTurn 1\n</message>\n\n<message role="assistant">\nResponse 1\n</message>\n\n<message role="user">\nTurn 2\n</message>\n\n',
      'Should send complete message history'
    );
  } finally {
    restore();
  }
});

test('non-stream chat completion returns OpenAI JSON instead of SSE', async () => {
  const restore = setupFetchMock((url) => {
    const stream = new ReadableStream({
      start(c) {
        c.enqueue(new TextEncoder().encode('data: {"p":"response/content","v":"hello"}\n\n'));
        c.enqueue(new TextEncoder().encode('data: {"p":"response/content","v":" world"}\n\n'));
        c.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
        c.close();
      }
    });
    return new Response(stream, { status: 200 });
  });

  try {
    const req = new Request('http://localhost/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'deepseek-v4-flash', messages: [{role: 'user', content: 'test'}], stream: false })
    });

    const res = await app.fetch(req);
    assert.strictEqual(res.status, 200);
    assert.match(res.headers.get('Content-Type') || '', /^application\/json/);

    const body = await res.json();
    assert.strictEqual(body.object, 'chat.completion');
    assert.strictEqual(body.choices[0].message.role, 'assistant');
    assert.strictEqual(body.choices[0].message.content, 'hello world');
    assert.strictEqual(body.choices[0].finish_reason, 'stop');
  } finally {
    restore();
  }
});


test('hermes-style XML tool calls are converted to structured OpenAI tool_calls', async () => {
  const restore = setupFetchMock((url) => {
    const stream = new ReadableStream({
      start(c) {
        c.enqueue(new TextEncoder().encode('data: {"p":"response/content","v":"Vou executar diretamente.\\n<tool_call><parameter name=\\"command\\">powershell.exe -Command Start-Process MuMuPlayer.exe</parameter><parameter name=\\"timeout\\">30</parameter></tool_call>"}\n\n'));
        c.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
        c.close();
      }
    });
    return new Response(stream, { status: 200 });
  });

  try {
    const req = new Request('http://localhost/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'deepseek-v4-flash',
        messages: [{ role: 'user', content: 'Abra o emulador MuMu' }],
        stream: false,
        tools: [{
          type: 'function',
          function: {
            name: 'terminal',
            description: 'Execute shell commands',
            parameters: {
              type: 'object',
              properties: { command: { type: 'string' }, timeout: { type: 'number' } },
              required: ['command']
            }
          }
        }]
      })
    });

    const res = await app.fetch(req);
    assert.strictEqual(res.status, 200);
    const body = await res.json();
    assert.strictEqual(body.choices[0].message.content, null);
    assert.strictEqual(body.choices[0].finish_reason, 'tool_calls');
    assert.strictEqual(body.choices[0].message.tool_calls[0].function.name, 'terminal');
    const args = JSON.parse(body.choices[0].message.tool_calls[0].function.arguments);
    assert.match(args.command, /MuMuPlayer\.exe/);
    assert.strictEqual(args.timeout, 30);
  } finally {
    restore();
  }
});

test('streaming Hermes-style XML tool calls do not leak as content', async () => {
  const restore = setupFetchMock((url) => {
    const stream = new ReadableStream({
      start(c) {
        c.enqueue(new TextEncoder().encode('data: {"p":"response/content","v":"<tool_call name=\\"terminal\\"><parameter name=\\"command\\">adb devices</parameter></tool_call>"}\n\n'));
        c.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
        c.close();
      }
    });
    return new Response(stream, { status: 200 });
  });

  try {
    const req = new Request('http://localhost/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'deepseek-v4-flash',
        messages: [{ role: 'user', content: 'liste adb' }],
        stream: true,
        tools: [{ type: 'function', function: { name: 'terminal', parameters: { type: 'object', properties: { command: { type: 'string' } } } } }]
      })
    });

    const res = await app.fetch(req);
    const text = await res.text();
    assert.ok(!text.includes('<tool_call'), 'tool XML must not leak into SSE content');
    assert.ok(!text.includes('<parameter'), 'parameter XML must not leak into SSE content');
    assert.ok(text.includes('"tool_calls"'), 'SSE must expose structured tool_calls');
    assert.ok(text.includes('"finish_reason":"tool_calls"'));
  } finally {
    restore();
  }
});

test('unclosed Hermes XML tool call at end of stream is recovered instead of returning empty', async () => {
  const restore = setupFetchMock((url) => {
    const stream = new ReadableStream({
      start(c) {
        c.enqueue(new TextEncoder().encode('data: {"p":"response/content","v":"<tool_call name=\\"terminal\\"><parameter name=\\"command\\">adb devices"}\n\n'));
        c.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
        c.close();
      }
    });
    return new Response(stream, { status: 200 });
  });

  try {
    const req = new Request('http://localhost/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'deepseek-v4-flash',
        messages: [{ role: 'user', content: 'liste adb' }],
        stream: true,
        tools: [{ type: 'function', function: { name: 'terminal', parameters: { type: 'object', properties: { command: { type: 'string' } } } } }]
      })
    });

    const res = await app.fetch(req);
    const text = await res.text();
    assert.ok(!text.includes('<tool_call'), 'unclosed tool XML must not leak into SSE content');
    assert.ok(!text.includes('<parameter'), 'unclosed parameter XML must not leak into SSE content');
    assert.ok(text.includes('"tool_calls"'), 'Recoverable unclosed XML must emit structured tool_calls');
    assert.ok(text.includes('adb devices'), 'Recovered tool call must preserve argument text');
    assert.ok(text.includes('"finish_reason":"tool_calls"'));
  } finally {
    restore();
  }
});

test('malformed internal tool call with lead-in returns safe content instead of empty response', async () => {
  const restore = setupFetchMock((url) => {
    const stream = new ReadableStream({
      start(c) {
        c.enqueue(new TextEncoder().encode('data: {"p":"response/content","v":"Não encontrei o dispositivo, vou verificar novamente.\\n<tool_call><parameter></parameter></tool_call>"}\n\n'));
        c.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
        c.close();
      }
    });
    return new Response(stream, { status: 200 });
  });

  try {
    const req = new Request('http://localhost/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'deepseek-v4-flash',
        messages: [{ role: 'user', content: 'continue depois de erro adb' }],
        stream: true,
        tools: [{ type: 'function', function: { name: 'terminal', parameters: { type: 'object', properties: { command: { type: 'string' } } } } }]
      })
    });

    const res = await app.fetch(req);
    const text = await res.text();
    assert.ok(!text.includes('<tool_call'), 'malformed tool XML must not leak into SSE content');
    assert.ok(!text.includes('<parameter'), 'malformed parameter XML must not leak into SSE content');
    assert.ok(!text.includes('"tool_calls"'), 'unparseable tool call must not be exposed as a fake structured tool call');
    assert.ok(text.includes('Não encontrei o dispositivo'), 'lead-in content must be preserved as a non-empty fallback');
    assert.ok(text.includes('"finish_reason":"stop"'));
  } finally {
    restore();
  }
});

test('streaming tool turn suppresses prose and transcript echo that arrives before tool_call', async () => {
  const restore = setupFetchMock((url) => {
    const stream = new ReadableStream({
      start(c) {
        c.enqueue(new TextEncoder().encode('data: {"p":"response/content","v":"That parsing approach is fragile.\\nTool Response (terminal): {\\\"output\\\":\\\"\\\"}\\nAssistant: <think>\\n"}\n\n'));
        c.enqueue(new TextEncoder().encode('data: {"p":"response/content","v":"<tool_call name=\\"terminal\\"><parameter name=\\"command\\">adb devices</parameter></tool_call>"}\n\n'));
        c.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
        c.close();
      }
    });
    return new Response(stream, { status: 200 });
  });

  try {
    const req = new Request('http://localhost/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'deepseek-v4-pro',
        messages: [{ role: 'user', content: 'debug adb' }],
        stream: true,
        tools: [{ type: 'function', function: { name: 'terminal', parameters: { type: 'object', properties: { command: { type: 'string' } } } } }]
      })
    });

    const res = await app.fetch(req);
    const text = await res.text();
    assert.ok(!text.includes('Tool Response (terminal)'), 'transcript echo must not leak into SSE content before a tool call');
    assert.ok(!text.includes('Assistant: <think>'), 'assistant transcript label must not leak into SSE content before a tool call');
    assert.ok(!text.includes('That parsing approach is fragile'), 'pre-tool explanatory prose should be suppressed for structured tool-call turns');
    assert.ok(text.includes('"tool_calls"'), 'SSE must expose the tool call structurally');
    assert.ok(text.includes('adb devices'), 'structured tool call should preserve arguments');
    assert.ok(text.includes('"finish_reason":"tool_calls"'));
  } finally {
    restore();
  }
});

test('JSON tool calls using tool-name shorthand are converted to structured OpenAI tool_calls', async () => {
  const restore = setupFetchMock((url) => {
    const stream = new ReadableStream({
      start(c) {
        c.enqueue(new TextEncoder().encode('data: {"p":"response/content","v":"Vou abrir o Chrome no dispositivo.\\n<tool_call>{\\"terminal\\":{\\"command\\":\\"adb -s 192.168.2.52:5555 shell uiautomator dump /sdcard/screen.xml\\",\\"timeout\\":30}}</tool_call>"}\n\n'));
        c.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
        c.close();
      }
    });
    return new Response(stream, { status: 200 });
  });

  try {
    const req = new Request('http://localhost/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'deepseek-v4-pro',
        messages: [{ role: 'user', content: 'use adb para consultar a tela' }],
        stream: false,
        tools: [{ type: 'function', function: { name: 'terminal', parameters: { type: 'object', properties: { command: { type: 'string' }, timeout: { type: 'number' } } } } }]
      })
    });

    const res = await app.fetch(req);
    assert.strictEqual(res.status, 200);
    const body = await res.json();
    assert.strictEqual(body.choices[0].message.content, null);
    assert.strictEqual(body.choices[0].finish_reason, 'tool_calls');
    assert.strictEqual(body.choices[0].message.tool_calls[0].function.name, 'terminal');
    const args = JSON.parse(body.choices[0].message.tool_calls[0].function.arguments);
    assert.strictEqual(args.command, 'adb -s 192.168.2.52:5555 shell uiautomator dump /sdcard/screen.xml');
    assert.strictEqual(args.timeout, 30);
  } finally {
    restore();
  }
});

test('excessive consecutive tool calls are capped to one per assistant turn by default', async () => {
  const restore = setupFetchMock((url) => {
    const stream = new ReadableStream({
      start(c) {
        c.enqueue(new TextEncoder().encode('data: {"p":"response/content","v":"<tool_call name=\\"terminal\\"><parameter name=\\"command\\">adb devices</parameter></tool_call><tool_call name=\\"terminal\\"><parameter name=\\"command\\">adb shell ss -ltnp</parameter></tool_call><tool_call name=\\"terminal\\"><parameter name=\\"command\\">curl localhost:9222/json/version</parameter></tool_call>"}\n\n'));
        c.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
        c.close();
      }
    });
    return new Response(stream, { status: 200 });
  });

  try {
    const req = new Request('http://localhost/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'deepseek-v4-pro',
        messages: [{ role: 'user', content: 'debug chrome adb' }],
        stream: false,
        tools: [{ type: 'function', function: { name: 'terminal', parameters: { type: 'object', properties: { command: { type: 'string' } } } } }]
      })
    });

    const res = await app.fetch(req);
    assert.strictEqual(res.status, 200);
    const body = await res.json();
    assert.strictEqual(body.choices[0].message.content, null);
    assert.strictEqual(body.choices[0].finish_reason, 'tool_calls');
    assert.strictEqual(body.choices[0].message.tool_calls.length, 1, 'only one tool call should be forwarded by default');
    const args = JSON.parse(body.choices[0].message.tool_calls[0].function.arguments);
    assert.strictEqual(args.command, 'adb devices');
  } finally {
    restore();
  }
});
