import { mkdir, readFile, unlink, writeFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import path from 'node:path'
import { MINI_CODE_SESSIONS_DIR } from './config.js'
import type { ChatMessage } from './types.js'

function sessionPath(cwd: string): string {
  const hash = createHash('md5').update(cwd).digest('hex')
  return path.join(MINI_CODE_SESSIONS_DIR, `${hash}.json`)
}

export async function saveSession(
  cwd: string,
  messages: ChatMessage[],
): Promise<void> {
  const toSave = messages.slice(1)
  if (toSave.length === 0) return
  const filePath = sessionPath(cwd)
  await mkdir(MINI_CODE_SESSIONS_DIR, { recursive: true })
  await writeFile(
    filePath,
    JSON.stringify({ cwd, messages: toSave, updatedAt: Date.now() }, null, 2),
    'utf8',
  )
}

export async function loadSession(
  cwd: string,
): Promise<ChatMessage[] | null> {
  try {
    const content = await readFile(sessionPath(cwd), 'utf8')
    const parsed = JSON.parse(content) as { messages?: unknown }
    if (!Array.isArray(parsed.messages)) return null
    return parsed.messages as ChatMessage[]
  } catch {
    return null
  }
}

export async function clearSession(cwd: string): Promise<void> {
  try {
    await unlink(sessionPath(cwd))
  } catch {
    // ignore ENOENT and other errors
  }
}
