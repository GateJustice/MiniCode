import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import type { ChatMessage } from '../src/types.js'
import {
  estimateMessageTokens,
  estimateMessagesTokens,
  computeContextStats,
} from '../src/utils/token-estimator.js'

describe('estimateMessageTokens', () => {
  it('estimates tokens for a system message', () => {
    const msg: ChatMessage = { role: 'system', content: 'You are a helpful assistant.' }
    const tokens = estimateMessageTokens(msg)
    assert.ok(tokens > 0)
    assert.ok(tokens < 100, `system message should be small, got ${tokens}`)
  })

  it('estimates tokens for a user message', () => {
    const msg: ChatMessage = { role: 'user', content: 'Hello, how are you?' }
    const tokens = estimateMessageTokens(msg)
    assert.ok(tokens > 0)
  })

  it('estimates more tokens for tool_result (higher density)', () => {
    const content = 'a'.repeat(100)
    const toolResult: ChatMessage = {
      role: 'tool_result',
      toolUseId: '1',
      toolName: 'read_file',
      content,
      isError: false,
    }
    const assistant: ChatMessage = {
      role: 'assistant',
      content,
    }
    const toolTokens = estimateMessageTokens(toolResult)
    const assistantTokens = estimateMessageTokens(assistant)
    assert.ok(toolTokens > assistantTokens, 'tool_result should estimate more tokens than assistant for same content length')
  })

  it('estimates tokens for assistant_tool_call with JSON input', () => {
    const msg: ChatMessage = {
      role: 'assistant_tool_call',
      toolUseId: '1',
      toolName: 'read_file',
      input: { path: '/some/long/path/to/file.ts' },
    }
    const tokens = estimateMessageTokens(msg)
    assert.ok(tokens > 0)
  })

  it('estimates tokens for context_summary', () => {
    const msg: ChatMessage = {
      role: 'context_summary',
      content: 'Summary of conversation so far.',
      compressedCount: 5,
      timestamp: Date.now(),
    }
    const tokens = estimateMessageTokens(msg)
    assert.ok(tokens > 0)
  })

  it('returns 0 for empty content', () => {
    const msg: ChatMessage = { role: 'user', content: '' }
    assert.equal(estimateMessageTokens(msg), 0)
  })
})

describe('estimateMessagesTokens', () => {
  it('sums tokens across all messages', () => {
    const messages: ChatMessage[] = [
      { role: 'system', content: 'System prompt here.' },
      { role: 'user', content: 'Hello!' },
      { role: 'assistant', content: 'Hi there!' },
    ]
    const total = estimateMessagesTokens(messages)
    const sum = messages.reduce((acc, msg) => acc + estimateMessageTokens(msg), 0)
    assert.equal(total, sum)
  })

  it('returns 0 for empty array', () => {
    assert.equal(estimateMessagesTokens([]), 0)
  })
})

describe('computeContextStats', () => {
  it('computes normal warning level for small messages', () => {
    const messages: ChatMessage[] = [
      { role: 'system', content: 'Hello' },
      { role: 'user', content: 'Test' },
    ]
    const stats = computeContextStats(messages, 'claude-sonnet-4-6')
    assert.equal(stats.warningLevel, 'normal')
    assert.ok(stats.utilization < 0.01, 'utilization should be very low')
    assert.ok(stats.estimatedTokens > 0)
    assert.equal(stats.contextWindow, 200_000)
    assert.equal(stats.effectiveInput, 184_000)
  })

  it('computes blocked warning level for large messages', () => {
    const messages: ChatMessage[] = [
      { role: 'system', content: 'x'.repeat(600_000) },
    ]
    const stats = computeContextStats(messages, 'deepseek-chat')
    assert.ok(
      stats.warningLevel === 'blocked' || stats.warningLevel === 'critical',
      `expected blocked or critical, got ${stats.warningLevel}`,
    )
    assert.equal(stats.utilization, 1, 'utilization should be capped at 1')
  })

  it('computes warning level for medium messages', () => {
    // effectiveInput for claude-sonnet-4-6 = 184000
    // 50% of 184000 = 92000 tokens
    // at ratio 3.5 for system, 92000 * 3.5 = 322000 chars
    const messages: ChatMessage[] = [
      { role: 'system', content: 'x'.repeat(160_000) },
      { role: 'user', content: 'x'.repeat(160_000) },
    ]
    const stats = computeContextStats(messages, 'claude-sonnet-4-6')
    assert.ok(
      stats.warningLevel === 'warning' || stats.warningLevel === 'critical',
      `expected warning or critical, got ${stats.warningLevel} (util: ${stats.utilization})`,
    )
    assert.ok(stats.utilization >= 0.5)
  })

  it('caps utilization at 1', () => {
    const messages: ChatMessage[] = [
      { role: 'user', content: 'x'.repeat(1_000_000) },
    ]
    const stats = computeContextStats(messages, 'deepseek-chat')
    assert.equal(stats.utilization, 1)
  })
})
