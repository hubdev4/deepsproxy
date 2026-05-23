import { appendFileSync, mkdirSync } from 'fs';
import { dirname, join, resolve } from 'path';
import { createHash } from 'crypto';

const truthy = /^(1|true|yes|y|on|debug)$/i;
const DEFAULT_LOG_DIR = resolve('logs');
const DEFAULT_LOG_FILE = 'deepsproxy-dev.log';

export function isDevLogEnabled(): boolean {
  return truthy.test(process.env.DEEPSPROXY_DEV_LOG || '') ||
    truthy.test(process.env.DEEPSPROXY_DEBUG || '') ||
    truthy.test(process.env.DEV_LOG || '') ||
    (process.env.LOG_LEVEL || '').toLowerCase() === 'debug';
}

function nowIso(): string {
  return new Date().toISOString();
}

export function shortHash(value: unknown): string {
  const text = typeof value === 'string' ? value : JSON.stringify(value ?? '');
  return createHash('sha256').update(text).digest('hex').slice(0, 16);
}

export const hashText = shortHash;

function redactString(value: string): string {
  return value
    .replace(/Bearer\s+[A-Za-z0-9._~+/=-]+/gi, 'Bearer [REDACTED]')
    .replace(/sk-[A-Za-z0-9._\-]+/g, 'sk-[REDACTED]')
    .replace(/(authorization\s*[:=]\s*)([^\s,;]+)/gi, '$1[REDACTED]')
    .replace(/(cookie\s*[:=]\s*)([^\n]+)/gi, '$1[REDACTED]')
    .replace(/(token=)[^;\s]+/gi, '$1[REDACTED]')
    .replace(/(password["'\s:=]+)[^,"'\s}]+/gi, '$1[REDACTED]')
    .replace(/(api[_-]?key["'\s:=]+)[^,"'\s}]+/gi, '$1[REDACTED]')
    .replace(/(x-ds-pow-response\s*[:=]\s*)([^\s,;]+)/gi, '$1[REDACTED]')
    .replace(/(x-hif-(?:dliq|leim)\s*[:=]\s*)([^\s,;]+)/gi, '$1[REDACTED]');
}

export function redact(value: unknown): unknown {
  if (value === null || value === undefined) return value;
  if (typeof value === 'string') return redactString(value);
  if (typeof value === 'number' || typeof value === 'boolean') return value;
  if (Array.isArray(value)) return value.map(redact);
  if (typeof value === 'object') {
    const out: Record<string, unknown> = {};
    for (const [key, val] of Object.entries(value as Record<string, unknown>)) {
      if (/cookie|authorization|password|token|api[_-]?key|secret|x-ds-pow-response|x-hif-(dliq|leim)/i.test(key)) {
        out[key] = val ? '[REDACTED]' : val;
      } else {
        out[key] = redact(val);
      }
    }
    return out;
  }
  return String(value);
}

export function snippet(text: unknown, max = 800): string {
  if (typeof text !== 'string') return '';
  const clean = redactString(text || '');
  return clean.length > max ? clean.slice(0, max) + `... [truncated ${clean.length - max} chars]` : clean;
}

export interface TraceContext {
  requestId: string;
  completionId?: string;
}

function logPath(): string {
  if (process.env.DEEPSPROXY_DEV_LOG_PATH) return resolve(process.env.DEEPSPROXY_DEV_LOG_PATH);
  if (process.env.DEEPSPROXY_DEBUG_LOG) return resolve(process.env.DEEPSPROXY_DEBUG_LOG);
  const dir = process.env.DEEPSPROXY_LOG_DIR || process.env.DEV_LOG_DIR || DEFAULT_LOG_DIR;
  return join(resolve(dir), DEFAULT_LOG_FILE);
}

export function getTraceLogPath(): string {
  return logPath();
}

export function devLog(event: string, data: Record<string, unknown> = {}) {
  if (!isDevLogEnabled()) return;
  const row = redact({
    ts: nowIso(),
    pid: process.pid,
    event,
    ...data,
  });
  try {
    const path = logPath();
    mkdirSync(dirname(path), { recursive: true });
    appendFileSync(path, JSON.stringify(row) + '\n', 'utf8');
  } catch (err) {
    // Logging must never break provider responses.
    console.error('[deepsproxy-dev-log] failed to write log:', err);
  }
}

export function trace(ctx: TraceContext | string | undefined, event: string, data: Record<string, unknown> = {}) {
  const requestId = typeof ctx === 'string' ? ctx : ctx?.requestId || 'no-request-id';
  const completionId = typeof ctx === 'string' ? undefined : ctx?.completionId;
  devLog(event, {
    requestId,
    completionId,
    ...data,
  });
}
