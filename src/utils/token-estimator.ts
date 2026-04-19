import type { ChatMessage } from '../types.js'
import { getModelContextWindow, type ModelContextWindow } from './model-context.js'

export type ContextStats = {
  estimatedTokens: number
  contextWindow: number
  effectiveInput: number
  utilization: number
  warningLevel: 'normal' | 'warning' | 'critical' | 'blocked'
}

const CHARS_PER_TOKEN: Record<string, number> = {
  system: 3.5,
  user: 3.0,
  assistant: 3.5,
  assistant_progress: 3.5,
  assistant_tool_call: 2.5,
  tool_result: 2.0,
  context_summary: 3.5,
}

const CLEAR_MARKER = '[Output cleared for context space]'

function messageContentLength(message: ChatMessage): number {
  switch (message.role) {
    case 'system':
    case 'user':
    case 'assistant':
    case 'assistant_progress':
      return message.content.length
    case 'assistant_tool_call':
      try {
        return JSON.stringify(message.input).length
      } catch {
        return 0
      }
    case 'tool_result':
      return message.content.length
    case 'context_summary':
      return message.content.length
    default:
      return 0
  }
}

export function estimateMessageTokens(message: ChatMessage): number {
  const ratio = CHARS_PER_TOKEN[message.role] ?? 3.0
  const length = messageContentLength(message)
  return Math.ceil(length / ratio)
}

export function estimateMessagesTokens(messages: ChatMessage[]): number {
  let total = 0
  for (const message of messages) {
    total += estimateMessageTokens(message)
  }
  return total
}

export function computeContextStats(
  messages: ChatMessage[],
  model: string,
): ContextStats {
  const window = getModelContextWindow(model)
  const estimatedTokens = estimateMessagesTokens(messages)
  const utilization = Math.min(1, estimatedTokens / window.effectiveInput)

  let warningLevel: ContextStats['warningLevel']
  if (utilization >= 0.95) {
    warningLevel = 'blocked'
  } else if (utilization >= 0.85) {
    warningLevel = 'critical'
  } else if (utilization >= 0.50) {
    warningLevel = 'warning'
  } else {
    warningLevel = 'normal'
  }

  return {
    estimatedTokens,
    contextWindow: window.contextWindow,
    effectiveInput: window.effectiveInput,
    utilization,
    warningLevel,
  }
}

export { CLEAR_MARKER }
