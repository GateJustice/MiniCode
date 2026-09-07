import type { PlanManager } from '../plan/manager.js'
import { GoalManager } from '../goal/manager.js'
import { GoalController } from '../goal/controller.js'
import { formatGoal, goalContext } from '../goal/context.js'
import { createGoalTools } from '../tools/goal.js'
import { ToolRegistry } from '../tool.js'
import type { AgentTurnResult } from '../types.js'
import { TurnRunner, type TurnRequest } from './turn-runner.js'

export class SessionRuntime {
  readonly turns = new TurnRunner()
  readonly goal: GoalController

  constructor(private readonly options: {
    plan: PlanManager
    tools: ToolRegistry
    execute: (request: TurnRequest) => Promise<AgentTurnResult>
    recordAnswer: (input: TurnRequest['input']) => Promise<void>
    notice: (message: string) => void
    settleWorkers: () => Promise<void>
    isBusy?: () => boolean
  }) {
    this.goal = new GoalController(new GoalManager(options.plan), {
      runTurn: request => this.run(request),
      recordAnswer: options.recordAnswer,
      notice: options.notice,
      settleWorkers: options.settleWorkers,
    })
  }

  toolsFor(mode?: TurnRequest['mode']): ToolRegistry {
    return mode === 'goal'
      ? new ToolRegistry([...this.options.tools.list(), ...createGoalTools(this.goal.manager)])
      : this.options.tools
  }

  contextFor(mode?: TurnRequest['mode']): string {
    return mode === 'goal' ? goalContext(this.goal.manager) : ''
  }

  async submit(content: string): Promise<void> {
    if (this.goal.waitingForAnswer) {
      await this.goal.answer(content)
      this.turns.awaitingUser = false
      return
    }
    if (this.goal.enabled) throw new Error('Goal is running. Use /goal pause before starting another turn.')
    await this.run({ input: { role: 'user', content } })
  }

  async command(input: string): Promise<string | null> {
    const match = /^\/goal(?:\s+([\s\S]*))?$/.exec(input.trim())
    if (!match) return null
    const body = match[1]?.trim() ?? ''
    const [command, ...rest] = body.split(/\s+/)
    const argument = rest.join(' ')
    if (!body || body === 'status') return formatGoal(this.goal.manager, this.goal.waitingForAnswer)
    if (command === 'pause') {
      await this.goal.pause(argument || 'Paused by user.')
    } else if (command === 'clear' && !argument) {
      const wasWaiting = this.goal.waitingForAnswer
      await this.goal.clear()
      if (wasWaiting) this.turns.awaitingUser = false
      return 'Goal cleared. Plan is retained.'
    } else if (command === 'resume' && !argument) {
      this.assertIdle()
      this.goal.resume()
    } else if (['update', 'edit', 'add', 'limit'].includes(command!)) {
      return 'This MVP keeps the original objective unchanged. Use /goal clear, then /goal <description>.'
    } else if (['status', 'resume', 'clear'].includes(command!)) {
      return `Usage: /goal ${command}`
    } else {
      this.assertIdle()
      this.goal.create(body)
    }
    return formatGoal(this.goal.manager, this.goal.waitingForAnswer)
  }

  async stop(): Promise<void> {
    await this.goal.pause('Execution stopped by user.')
    await this.turns.stop()
    await this.options.settleWorkers()
  }

  async reset(): Promise<void> {
    await this.stop()
    await this.goal.clear()
    this.turns.awaitingUser = false
  }

  private assertIdle(): void {
    if (this.turns.busy || this.goal.running || this.options.isBusy?.()) throw new Error('Wait for the current turn to stop.')
    if (this.turns.awaitingUser && !this.goal.waitingForAnswer) throw new Error('Answer the pending question first.')
  }

  private run(request: TurnRequest): Promise<AgentTurnResult> {
    return this.turns.run(signal => this.options.execute({ ...request, signal }), request.signal)
  }
}
