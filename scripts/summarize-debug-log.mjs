#!/usr/bin/env node
import fs from 'fs';

const path = process.argv[2] || 'logs/deepsproxy-dev.log';
const rows = fs.existsSync(path)
  ? fs.readFileSync(path, 'utf8').trim().split('\n').filter(Boolean).map(l => {
      try { return JSON.parse(l); } catch { return { event: 'parse_error', raw: l }; }
    })
  : [];
const byReq = new Map();
for (const row of rows) {
  const id = row.requestId || 'no-request-id';
  if (!byReq.has(id)) byReq.set(id, []);
  byReq.get(id).push(row);
}
const requests = [...byReq.entries()].slice(-(Number(process.argv[3]) || 5));
for (const [id, events] of requests) {
  const first = events[0];
  const last = events[events.length - 1];
  const req = events.find(e => e.event === 'request_start') || {};
  const parse = [...events].reverse().find(e => e.event === 'parse_complete') || {};
  const ready = [...events].reverse().find(e => e.event === 'response_nonstream_ready' || e.event === 'response_stream_complete') || {};
  const err = [...events].reverse().find(e => /error|timeout|failed/.test(e.event)) || null;
  console.log(`\nrequestId=${id}`);
  console.log(`  time=${first?.ts}..${last?.ts} events=${events.length}`);
  console.log(`  model=${req.model} stream=${req.stream} messages=${req.messagesCount} tools=${req.toolsCount} promptTokens=${req.promptTokens}`);
  console.log(`  parse: contentLength=${parse.contentLength} reasoningLength=${parse.reasoningLength} toolCalls=${parse.toolCallsCount} finish=${parse.finishReason} rawChunks=${parse.rawChunkCount} foundText=${parse.foundTextChunkCount} ignored=${parse.ignoredNoTextCount} parseErrors=${parse.parseErrorCount}`);
  console.log(`  response: event=${ready.event} contentLength=${ready.contentLength} toolCalls=${ready.toolCallsCount} finish=${ready.finishReason} elapsedMs=${ready.elapsedMs}`);
  if (ready.contentSnippet) console.log(`  snippet=${ready.contentSnippet}`);
  if (err) console.log(`  last_error_event=${err.event} message=${err.message || ''}`);
  console.log(`  last_events=${events.slice(-8).map(e => e.event).join(' -> ')}`);
}
