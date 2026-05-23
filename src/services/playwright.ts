/*
 * File: playwright.ts
 * Project: deepsproxy
 * Author: Pedro Farias
 * Created: 2026-05-09
 * 
 * Last Modified: Sat May 09 2026
 * Modified By: Pedro Farias
 */

import { chromium, BrowserContext, Page } from 'playwright';
import path from 'path';
import { shortHash, snippet, trace, TraceContext } from '../utils/trace.ts';

let context: BrowserContext | null = null;
export let activePage: Page | null = null;
let currentHeaders: Record<string, string> = {};

export async function initPlaywright(headless = true, ctx?: TraceContext) {
  if (process.env.TEST_MOCK_PLAYWRIGHT) return;

  trace(ctx, 'playwright_init_start', { headless, hasContext: Boolean(context), hasActivePage: Boolean(activePage) });

  // If context exists but browser is dead, clean up first
  if (context) {
    try {
      // Test if browser is still alive by checking if we can access it
      if (activePage) {
        await activePage.url(); // throws if browser is dead
        trace(ctx, 'playwright_init_reuse_existing', { url: activePage.url() });
        return; // browser is alive, nothing to do
      }
    } catch {
      console.log('[playwright] Existing context is dead, cleaning up before reinit...');
    }
    try { await context.close(); } catch {}
    context = null;
    activePage = null;
  }

  const profilePath = path.resolve('deepseek_profile');

  context = await chromium.launchPersistentContext(profilePath, {
    headless,
    userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
    args: [
      '--disable-blink-features=AutomationControlled',
      '--exclude-switches=enable-automation',
      '--disable-infobars',
      '--no-sandbox',
      '--disable-dev-shm-usage',
      '--disable-gpu',
    ],
  });

  // Keep an active page to fetch PoW headers on demand
  activePage = await context.newPage();
  trace(ctx, 'playwright_init_complete', { profilePath, pages: context.pages().length, activeUrl: activePage.url() });
}

export async function closePlaywright() {
  if (process.env.TEST_MOCK_PLAYWRIGHT) return;
  if (context) {
    await context.close();
    context = null;
    activePage = null;
  }
}

/**
 * Ensures the session is valid and extracts headers, PoW, and session ID.
 */
export async function getDeepSeekHeaders(forceNew = false, ctx?: TraceContext): Promise<{ headers: Record<string, string>, chatSessionId: string, parentMessageId: number | null }> {
  if (process.env.TEST_MOCK_PLAYWRIGHT) {
    // Generate a unique session ID if requested for testing isolation
    const mockSessionId = process.env.TEST_SESSION_ID || 'mock-session';
    return { headers: { authorization: 'Bearer MOCK' }, chatSessionId: mockSessionId, parentMessageId: null };
  }

  // Auto-recover if browser crashed
  trace(ctx, 'headers_start', { forceNew, hasContext: Boolean(context), hasActivePage: Boolean(activePage), currentUrl: activePage ? activePage.url() : null });
  if (!activePage || !context) {
    console.log('[playwright] Browser not available, reinitializing...');
    trace(ctx, 'headers_reinit_browser_missing', {});
    await initPlaywright(true, ctx);
    if (!activePage) {
      throw new Error('Playwright failed to reinitialize');
    }
  }

  // Navigate to deepseek chat with crash recovery
  try {
    const currentUrl = activePage.url();
    const isOnDeepSeek = currentUrl.includes('chat.deepseek.com');
    const isOnSpecificChat = isOnDeepSeek && /\/chat\/\d+/.test(currentUrl);

    if (!isOnDeepSeek || forceNew || isOnSpecificChat) {
      trace(ctx, 'headers_navigate_start', { currentUrl, forceNew, isOnDeepSeek, isOnSpecificChat });
      const navStart = Date.now();
      await activePage.goto('https://chat.deepseek.com/', { waitUntil: 'domcontentloaded', timeout: 15000 });
      trace(ctx, 'headers_navigate_done', { elapsedMs: Date.now() - navStart, url: activePage.url(), title: await activePage.title().catch(() => '') });
    } else {
      trace(ctx, 'headers_navigate_skipped', { currentUrl });
    }
  } catch (navError: any) {
    console.error('[playwright] Navigation failed, attempting browser recovery:', navError.message);
    trace(ctx, 'headers_navigate_error_recovering', { message: navError?.message || String(navError), stack: navError?.stack });
    try { await closePlaywright(); } catch {}
    await initPlaywright(true, ctx);
    if (!activePage) {
      throw new Error('Playwright recovery failed: no active page');
    }
    await activePage.goto('https://chat.deepseek.com/', { waitUntil: 'domcontentloaded', timeout: 30000 });
  }

  // Wait for the chat input. Keep this timeout short: when DeepSeek shows an
  // account/login/suspension banner there is no input, and retrying the same
  // browser state just makes OpenAI clients look hung.
  const chatInputSelector = 'textarea, [role="textbox"], [contenteditable="true"]';
  const chatInputTimeoutMs = Number(process.env.DEEPSPROXY_CHAT_INPUT_TIMEOUT_MS || '8000');
  const textareaStart = Date.now();
  await activePage.waitForSelector(chatInputSelector, { timeout: chatInputTimeoutMs }).then(async () => {
    trace(ctx, 'headers_textarea_ready', { elapsedMs: Date.now() - textareaStart, url: activePage!.url(), title: await activePage!.title().catch(() => '') });
  }).catch(async () => {
    const pageState = await activePage!.evaluate(() => {
      const fullBodyText = document.body?.innerText || '';
      const bodyText = fullBodyText.slice(0, 5000);
      const suspensionMatch = fullBodyText.match(/Due to violation of user policies, your account has been suspended until\s+([^\.\n]+)\.\s*If you have any questions, please Contact us\./i);
      const suspendedUntil = suspensionMatch?.[1]?.trim() || null;
      const suspensionOriginal = suspensionMatch?.[0]?.trim() || null;
      return {
        url: location.href,
        title: document.title,
        bodyText,
        textareaCount: document.querySelectorAll('textarea').length,
        inputCount: document.querySelectorAll('input, textarea, [role="textbox"], [contenteditable]').length,
        suspended: /suspended until|violation of user policies|account has been suspended/i.test(fullBodyText),
        suspendedUntil,
        suspensionOriginal,
        loginRequired: /log in|login|sign in|entrar/i.test(fullBodyText),
      };
    }).catch((e: any) => ({ evaluateError: e?.message || String(e) }));
    trace(ctx, 'headers_textarea_timeout', { elapsedMs: Date.now() - textareaStart, pageState });

    const state: any = pageState;
    if (state?.suspended) {
      const until = typeof state.suspendedUntil === 'string' && state.suspendedUntil.trim() ? state.suspendedUntil.trim() : '';
      const original = typeof state.suspensionOriginal === 'string' && state.suspensionOriginal.trim() ? state.suspensionOriginal.trim() : '';
      const detail = original || (until ? `Due to violation of user policies, your account has been suspended until ${until}.` : 'DeepSeek reported an account suspension.');
      throw new Error(`DeepSeek account is suspended; chat input is unavailable. Original DeepSeek message: ${detail}`);
    }
    if (state?.loginRequired) {
      throw new Error('DeepSeek login is required; chat input is unavailable.');
    }
    throw new Error('DeepSeek chat input unavailable; page did not expose an input box.');
  });

  return new Promise((resolve, reject) => {
    const powStart = Date.now();
    const timeout = setTimeout(async () => {
      const pageState = await activePage!.evaluate(() => ({
        url: location.href,
        title: document.title,
        bodyText: (document.body?.innerText || '').slice(0, 1200),
        textareaValueLength: (document.querySelector('textarea') as HTMLTextAreaElement | null)?.value?.length ?? null,
      })).catch((e: any) => ({ evaluateError: e?.message || String(e) }));
      trace(ctx, 'headers_pow_timeout', { elapsedMs: Date.now() - powStart, pageState });
      reject(new Error('Timeout waiting for PoW headers'));
    }, 30000);

    const routeHandler = async (route: any, request: any) => {
      clearTimeout(timeout);
      
      const reqHeaders = request.headers();
      let uiSessionId = '';
      let uiParentMessageId: number | null = null;

      const postData = request.postData();
      if (postData) {
        try {
          const payload = JSON.parse(postData);
          if (payload.chat_session_id) {
            uiSessionId = payload.chat_session_id;
          }
          if (payload.parent_message_id !== undefined) {
            uiParentMessageId = payload.parent_message_id;
          }
        } catch (e) {
          // ignore parsing error
        }
      }

      const extractedHeaders = {
        'x-ds-pow-response': reqHeaders['x-ds-pow-response'] || '',
        'x-hif-dliq': reqHeaders['x-hif-dliq'] || '',
        'x-hif-leim': reqHeaders['x-hif-leim'] || '',
        'authorization': reqHeaders['authorization'] || '',
        'cookie': reqHeaders['cookie'] || ''
      };

      currentHeaders = extractedHeaders;
      trace(ctx, 'headers_route_captured', {
        elapsedMs: Date.now() - powStart,
        url: request.url(),
        method: request.method(),
        postDataLength: postData?.length || 0,
        postDataHash: shortHash(postData || ''),
        postDataSnippet: snippet(postData || '', 300),
        uiSessionIdHash: shortHash(uiSessionId),
        uiParentMessageId,
        hasAuthorization: Boolean(extractedHeaders.authorization),
        hasCookie: Boolean(extractedHeaders.cookie),
        hasPow: Boolean(extractedHeaders['x-ds-pow-response']),
        hasHifDliq: Boolean(extractedHeaders['x-hif-dliq']),
        hasHifLeim: Boolean(extractedHeaders['x-hif-leim'])
      });

      // Abort to prevent polluting chat history
      await route.abort('aborted');
      
      // Cleanup route
      await activePage!.unroute('**/api/v0/chat/completion', routeHandler);

      resolve({ headers: extractedHeaders, chatSessionId: uiSessionId, parentMessageId: uiParentMessageId });
    };

    activePage!.route('**/api/v0/chat/completion', routeHandler).then(() => {
      // Trigger PoW generation by typing and hitting enter
      trace(ctx, 'headers_route_installed_trigger_pow', {});
      activePage!.fill('textarea', 'a').then(() => {
        trace(ctx, 'headers_pow_trigger_filled', {});
        activePage!.keyboard.press('Enter').then(() => {
          trace(ctx, 'headers_pow_trigger_enter_pressed', {});
        }).catch((e: any) => {
          trace(ctx, 'headers_pow_trigger_enter_error', { message: e?.message || String(e), stack: e?.stack });
          reject(e);
        });
      }).catch((e: any) => {
        trace(ctx, 'headers_pow_trigger_fill_error', { message: e?.message || String(e), stack: e?.stack });
        reject(e);
      });
    }).catch((e: any) => {
      trace(ctx, 'headers_route_install_error', { message: e?.message || String(e), stack: e?.stack });
      reject(e);
    });
  });
}
