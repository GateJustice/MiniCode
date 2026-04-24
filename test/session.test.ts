import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert/strict'
import { mkdir, rm } from 'node:fs/promises'
import path from 'node:path'
import os from 'node:os'
import { saveSession, loadSession, clearSession } from '../src/session.js'
import type { ChatMessage } from '../src/types.js'

const testDir = path.join(os.tmpdir(), 'minicode-session-test')

function makeMessages(count: number): ChatMessage[] {
  const messages: ChatMessage[] = [
    { role: 'system', content: 'You are a helpful assistant.' },
  ]
  for (let i = 0; i < count; i++) {
    messages.push({ role: 'user', content: `User message ${i}` })
    messages.push({ role: 'assistant', content: `Assistant response ${i}` })
  }
  return messages
}

describe('session persistence', () => {
  beforeEach(async () => {
    await mkdir(testDir, { recursive: true })
  })

  afterEach(async () => {
    await rm(testDir, { recursive: true, force: true })
  })

  it('round-trips messages and excludes system prompt', async () => {
    const cwd = path.join(testDir, 'project-a')
    const messages = makeMessages(3)

    await saveSession(cwd, messages)
    const loaded = await loadSession(cwd)

    assert.notEqual(loaded, null)
    assert.equal(loaded!.length, 6)
    assert.equal(loaded![0].role, 'user')
    assert.equal(loaded![0].content, 'User message 0')
    assert.equal(loaded![5].role, 'assistant')
    assert.equal(loaded![5].content, 'Assistant response 2')
  })

  it('returns null for nonexistent session', async () => {
    const cwd = path.join(testDir, 'no-such-project')
    const loaded = await loadSession(cwd)
    assert.equal(loaded, null)
  })

  it('clears an existing session', async () => {
    const cwd = path.join(testDir, 'project-b')
    await saveSession(cwd, makeMessages(1))
    assert.notEqual(await loadSession(cwd), null)

    await clearSession(cwd)
    assert.equal(await loadSession(cwd), null)
  })

  it('clearSession does not throw for nonexistent session', async () => {
    const cwd = path.join(testDir, 'no-such-project')
    await assert.doesNotReject(() => clearSession(cwd))
  })

  it('saveSession skips save when only system prompt exists', async () => {
    const cwd = path.join(testDir, 'empty-project')
    await saveSession(cwd, [{ role: 'system', content: 'system' }])
    assert.equal(await loadSession(cwd), null)
  })

  it('preserves all ChatMessage role types', async () => {
    const cwd = path.join(testDir, 'all-types')
    const messages: ChatMessage[] = [
      { role: 'system', content: 'sys' },
      { role: 'user', content: 'hello' },
      { role: 'assistant', content: 'hi' },
      { role: 'assistant_tool_call', toolUseId: 'c1', toolName: 'read_file', input: { path: '/a.ts' } },
      { role: 'tool_result', toolUseId: 'c1', toolName: 'read_file', content: 'file contents', isError: false },
      { role: 'context_summary', content: 'summary text', compressedCount: 5, timestamp: 12345 },
    ]

    await saveSession(cwd, messages)
    const loaded = await loadSession(cwd)

    assert.equal(loaded!.length, 5)
    assert.equal(loaded![0].role, 'user')
    assert.equal(loaded![1].role, 'assistant')
    assert.equal(loaded![2].role, 'assistant_tool_call')
    assert.equal(loaded![3].role, 'tool_result')
    assert.equal(loaded![4].role, 'context_summary')
  })
})
