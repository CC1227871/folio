import { randomUUID } from 'node:crypto';
import type {
  AgentEvent,
  AgentEventPayload,
  AgentRuntime,
  ApiError,
  ConversationBranch,
  ConversationOperation,
  Message,
  Run,
  RunContextSnapshot,
  RuntimeRunArtifacts,
  SessionMeta,
  ToolCall,
  ToolCallRecord,
  WorkspaceContext,
  SupportedLocale,
} from '@finagent/core';
import type { RunRepository } from '../storage/index.ts';
import type { SessionManager } from './session-manager.ts';
import { createCodeError, isRuntimeInfraCode, toApiError } from '../agent/errors.ts';

export interface RunManagerOptions {
  sessions: SessionManager;
  runs: RunRepository;
  runtime: AgentRuntime;
  now?: () => number;
}

interface ActiveRun {
  sessionId: string;
  runId: string;
  cancelRequested: boolean;
}

interface GenerationOptions {
  session: SessionMeta;
  branch: ConversationBranch;
  content: string;
  operation: ConversationOperation;
  sourceMessageId?: string;
  parentRunId?: string;
  parentMessageId?: string;
  existingUserMessage?: Message;
  workspaceContext?: WorkspaceContext;
  locale?: SupportedLocale;
}

/**
 * Starts, observes, persists, and terminates runs.
 *
 * Each run: persists the user message and the run record, drives the runtime's
 * event stream, broadcasts every AgentEvent to subscribers (the Electron main
 * process forwards them to the UI), persists the final assistant message and
 * run outcome, and guarantees the run never stays `running`: every terminal
 * path (completed, failed, cancelled, runtime crash, timeout) settles it.
 */
export class RunManager {
  private readonly sessions: SessionManager;
  private readonly runs: RunRepository;
  private readonly runtime: AgentRuntime;
  private readonly now: () => number;
  private readonly listeners = new Set<(event: AgentEvent) => void>();
  private activeRun: ActiveRun | null = null;

  constructor(options: RunManagerOptions) {
    this.sessions = options.sessions;
    this.runs = options.runs;
    this.runtime = options.runtime;
    this.now = options.now ?? Date.now;
  }

  subscribe(listener: (event: AgentEvent) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** Whether a run is currently executing (Pi runtime executes one at a time). */
  isRunning(): boolean {
    return this.activeRun !== null;
  }

  /** Whether a run is currently executing for the given session. */
  hasActiveRun(sessionId: string): boolean {
    return this.activeRun?.sessionId === sessionId;
  }

  async startRun(
    sessionId: string,
    content: string,
    workspaceContext?: WorkspaceContext,
    locale?: SupportedLocale
  ): Promise<Run> {
    const session = await this.sessions.getSession(sessionId);
    if (!session) {
      throw createCodeError('SESSION_NOT_FOUND', `Session ${sessionId} was not found.`);
    }
    const branch = await this.sessions.getActiveBranch(sessionId);
    if (!branch) {
      throw createCodeError('BRANCH_NOT_FOUND', `The active branch for session ${sessionId} was not found.`);
    }
    const visibleMessages = await this.sessions.listMessages(sessionId, branch.id);
    return this.startGeneration({
      session,
      branch,
      content,
      operation: 'send',
      parentMessageId: visibleMessages.at(-1)?.id,
      workspaceContext,
      locale,
    });
  }

  /** Retry a failed/cancelled generation on a new child branch. */
  async retryRun(
    sessionId: string,
    runId: string,
    workspaceContext?: WorkspaceContext,
    locale?: SupportedLocale
  ): Promise<Run> {
    this.assertNoActiveRun();
    const sourceRun = await this.sessions.getRun(sessionId, runId);
    if (!sourceRun) throw createCodeError('RUN_NOT_FOUND', `Run ${runId} was not found.`);
    if (sourceRun.status !== 'failed' && sourceRun.status !== 'cancelled') {
      throw createCodeError('INVALID_ARGUMENT', 'Only failed or cancelled runs can be retried.');
    }

    const sourceBranch = await this.branchForRun(sessionId, sourceRun);
    const visible = await this.sessions.listMessages(sessionId, sourceBranch.id);
    const userIndex = findUserMessageIndex(visible, sourceRun);
    const sourceUser = userIndex >= 0 ? visible[userIndex] : undefined;
    const runtimeForkEntryId = sourceRun.runtimeUserEntryId ??
      (sourceUser ? await this.runtimeForkEntryForMessage(sessionId, sourceUser) : undefined);
    const child = await this.createDerivedBranch(
      sessionId,
      sourceBranch,
      userIndex > 0 ? visible[userIndex - 1].id : null,
      `Retry · ${sourceRun.input.slice(0, 32)}`,
      runtimeForkEntryId
    );
    const session = await this.requireSession(sessionId);
    return this.startGeneration({
      session,
      branch: child,
      content: sourceRun.input,
      operation: 'retry',
      sourceMessageId: sourceUser?.id,
      parentRunId: sourceRun.id,
      parentMessageId: userIndex > 0 ? visible[userIndex - 1].id : undefined,
      workspaceContext,
      locale,
    });
  }

  /** Regenerate an assistant artifact while inheriting the same user context. */
  async regenerateMessage(
    sessionId: string,
    assistantMessageId: string,
    workspaceContext?: WorkspaceContext,
    locale?: SupportedLocale
  ): Promise<Run> {
    this.assertNoActiveRun();
    const sourceBranch = await this.sessions.getActiveBranch(sessionId);
    if (!sourceBranch) throw createCodeError('BRANCH_NOT_FOUND', 'The active branch was not found.');
    const visible = await this.sessions.listMessages(sessionId, sourceBranch.id);
    const assistant = visible.find((message) => message.id === assistantMessageId);
    if (!assistant || assistant.role !== 'assistant') {
      throw createCodeError('MESSAGE_NOT_FOUND', `Assistant message ${assistantMessageId} was not found.`);
    }
    const sourceRun = assistant.runId ? await this.sessions.getRun(sessionId, assistant.runId) : null;
    if (!sourceRun) {
      throw createCodeError('RUN_NOT_FOUND', `The generation for message ${assistantMessageId} was not found.`);
    }
    const userMessage = sourceRun.userMessageId
      ? visible.find((message) => message.id === sourceRun.userMessageId)
      : findPreviousUserMessage(visible, visible.indexOf(assistant));
    if (!userMessage) {
      throw createCodeError('MESSAGE_NOT_FOUND', 'The user context for this answer was not found.');
    }
    const runtimeForkEntryId = sourceRun.runtimeUserEntryId ??
      await this.runtimeForkEntryForMessage(sessionId, userMessage);
    const child = await this.createDerivedBranch(
      sessionId,
      sourceBranch,
      userMessage.id,
      `Regenerate · ${userMessage.content.slice(0, 28)}`,
      runtimeForkEntryId
    );
    const session = await this.requireSession(sessionId);
    return this.startGeneration({
      session,
      branch: child,
      content: userMessage.content,
      operation: 'regenerate',
      sourceMessageId: assistant.id,
      parentRunId: sourceRun.id,
      parentMessageId: userMessage.id,
      existingUserMessage: userMessage,
      workspaceContext,
      locale,
    });
  }

  /** Edit a historical user message and rerun from the preceding cursor. */
  async editMessage(
    sessionId: string,
    userMessageId: string,
    content: string,
    workspaceContext?: WorkspaceContext,
    locale?: SupportedLocale
  ): Promise<Run> {
    this.assertNoActiveRun();
    const text = content.trim();
    if (!text) throw createCodeError('INVALID_ARGUMENT', 'Message content is required.');
    const sourceBranch = await this.sessions.getActiveBranch(sessionId);
    if (!sourceBranch) throw createCodeError('BRANCH_NOT_FOUND', 'The active branch was not found.');
    const visible = await this.sessions.listMessages(sessionId, sourceBranch.id);
    const sourceIndex = visible.findIndex((message) => message.id === userMessageId);
    const sourceMessage = sourceIndex >= 0 ? visible[sourceIndex] : undefined;
    if (!sourceMessage || sourceMessage.role !== 'user') {
      throw createCodeError('MESSAGE_NOT_FOUND', `User message ${userMessageId} was not found.`);
    }
    const runtimeForkEntryId = await this.runtimeForkEntryForMessage(sessionId, sourceMessage);
    const child = await this.createDerivedBranch(
      sessionId,
      sourceBranch,
      sourceIndex > 0 ? visible[sourceIndex - 1].id : null,
      `Edit · ${text.slice(0, 32)}`,
      runtimeForkEntryId
    );
    const session = await this.requireSession(sessionId);
    return this.startGeneration({
      session,
      branch: child,
      content: text,
      operation: 'edit',
      sourceMessageId: sourceMessage.id,
      parentMessageId: sourceIndex > 0 ? visible[sourceIndex - 1].id : undefined,
      workspaceContext,
      locale,
    });
  }

  /** Create and activate a branch from an earlier visible message. */
  async forkBranch(sessionId: string, messageId: string, name?: string): Promise<ConversationBranch> {
    this.assertNoActiveRun();
    const sourceBranch = await this.sessions.getActiveBranch(sessionId);
    if (!sourceBranch) throw createCodeError('BRANCH_NOT_FOUND', 'The active branch was not found.');
    const visible = await this.sessions.listMessages(sessionId, sourceBranch.id);
    if (!visible.some((message) => message.id === messageId)) {
      throw createCodeError('MESSAGE_NOT_FOUND', `Message ${messageId} was not found.`);
    }
    const sourceMessage = visible.find((message) => message.id === messageId);
    const runtimeForkEntryId = sourceMessage
      ? await this.runtimeForkEntryForMessage(sessionId, sourceMessage)
      : undefined;
    const child = await this.createDerivedBranch(
      sessionId,
      sourceBranch,
      messageId,
      name?.trim() || `Fork · ${new Date(this.now()).toISOString()}`,
      runtimeForkEntryId
    );
    await this.sessions.setActiveBranch(sessionId, child.id);
    return child;
  }

  async setActiveBranch(sessionId: string, branchId: string): Promise<ConversationBranch> {
    this.assertNoActiveRun();
    const branch = await this.sessions.setActiveBranch(sessionId, branchId);
    if (!branch) throw createCodeError('BRANCH_NOT_FOUND', `Branch ${branchId} was not found.`);
    return branch;
  }

  /** Abort the given run if it is the one currently executing. */
  async cancelRun(sessionId: string, runId: string): Promise<void> {
    const active = this.activeRun;
    if (!active || active.sessionId !== sessionId || active.runId !== runId) {
      return;
    }
    active.cancelRequested = true;
    await this.runtime.cancel({ sessionId, runId });
  }

  private async startGeneration(options: GenerationOptions): Promise<Run> {
    const text = options.content.trim();
    if (!text) throw createCodeError('INVALID_ARGUMENT', 'Message content is required.');
    if (this.activeRun) {
      throw createCodeError(
        'RUN_IN_PROGRESS',
        'Another run is still in progress. Stop it before sending a new message.'
      );
    }

    const now = this.now();
    const runId = randomUUID();
    const contextSnapshot: RunContextSnapshot = {
      capturedAt: now,
      workspaceContext: options.workspaceContext,
      recentSymbols: options.session.recentSymbols ? [...options.session.recentSymbols] : undefined,
    };
    const userMessage = options.existingUserMessage ?? {
      id: randomUUID(),
      role: 'user' as const,
      content: text,
      timestamp: now,
      branchId: options.branch.id,
      parentMessageId: options.parentMessageId,
      runId,
      generationId: runId,
      operation: options.operation,
      sourceMessageId: options.sourceMessageId,
      contextSnapshot,
    };
    const run: Run = {
      id: runId,
      sessionId: options.session.id,
      status: 'running',
      input: text,
      startedAt: now,
      branchId: options.branch.id,
      operation: options.operation,
      parentRunId: options.parentRunId,
      sourceMessageId: options.sourceMessageId,
      userMessageId: userMessage.id,
      generationId: runId,
      contextSnapshot,
      manifest: {
        runId,
        branchId: options.branch.id,
        operation: options.operation,
        inputMessageId: userMessage.id,
        parentRunId: options.parentRunId,
        sourceMessageId: options.sourceMessageId,
        contextSnapshot,
        toolCallIds: [],
      },
    };

    await this.runs.create(run);
    if (!options.existingUserMessage) {
      await this.sessions.appendMessage(options.session.id, userMessage);
    }
    await this.sessions.updateSession(options.session.id, { status: 'running' });

    this.activeRun = { sessionId: options.session.id, runId: run.id, cancelRequested: false };
    this.emit({
      id: randomUUID(),
      sessionId: options.session.id,
      runId: run.id,
      type: 'run_started',
      timestamp: now,
      sequence: 1,
      payload: {
        run,
        userMessage,
        userMessageIsNew: !options.existingUserMessage,
      },
    });

    void this.execute(run, options.session, options.workspaceContext, options.locale);
    return run;
  }

  private assertNoActiveRun(): void {
    if (this.activeRun) {
      throw createCodeError(
        'RUN_IN_PROGRESS',
        'Another run is still in progress. Stop it before changing conversation branches.'
      );
    }
  }

  private async requireSession(sessionId: string): Promise<SessionMeta> {
    const session = await this.sessions.getSession(sessionId);
    if (!session) throw createCodeError('SESSION_NOT_FOUND', `Session ${sessionId} was not found.`);
    return session;
  }

  private async branchForRun(sessionId: string, run: Run): Promise<ConversationBranch> {
    const branch = run.branchId
      ? await this.sessions.getBranch(sessionId, run.branchId)
      : await this.sessions.getActiveBranch(sessionId);
    if (!branch) throw createCodeError('BRANCH_NOT_FOUND', 'The source branch was not found.');
    return branch;
  }

  /** Resolve the native Pi user-entry cursor for a Folio message. */
  private async runtimeForkEntryForMessage(sessionId: string, message: Message): Promise<string | undefined> {
    const run = message.runId ? await this.sessions.getRun(sessionId, message.runId) : null;
    if (message.role === 'user') {
      return run?.runtimeUserEntryId ?? message.runtimeEntryId;
    }
    // Pi forks before a user entry. When the UI forks from an assistant
    // artifact, use the user generation that produced that artifact.
    if (message.role === 'assistant') {
      return run?.runtimeUserEntryId;
    }
    return message.runtimeEntryId;
  }

  private async createDerivedBranch(
    sessionId: string,
    parent: ConversationBranch,
    forkMessageId: string | null | undefined,
    name: string,
    runtimeForkEntryId?: string
  ): Promise<ConversationBranch> {
    const child = await this.sessions.createBranch({
      sessionId,
      name,
      parentBranchId: parent.id,
      forkMessageId,
      runtimeForkEntryId,
    });
    await this.sessions.setActiveBranch(sessionId, child.id);
    return child;
  }

  private async execute(
    run: Run,
    session: SessionMeta,
    workspaceContext?: WorkspaceContext,
    locale?: SupportedLocale
  ): Promise<void> {
    let failure: ApiError | undefined;
    let answer = '';
    const toolCalls: ToolCall[] = [];
    let sawTerminal = false;

    try {
      let branch = run.branchId
        ? await this.sessions.getBranch(run.sessionId, run.branchId)
        : null;
      let runtimeSessionPath = branch?.runtimeSessionPath ?? session.runtimeSessionPath;

      if (branch?.parentBranchId && !branch.runtimeSessionPath && this.runtime.prepareBranch) {
        const parent = await this.sessions.getBranch(run.sessionId, branch.parentBranchId);
        const prepared = await this.runtime.prepareBranch({
          sessionId: run.sessionId,
          branchId: branch.id,
          parentBranchId: branch.parentBranchId,
          parentSessionPath: parent?.runtimeSessionPath ?? session.runtimeSessionPath,
          forkMessageId: branch.forkMessageId,
          forkRuntimeEntryId: branch.runtimeForkEntryId,
        });
        branch = await this.sessions.updateBranch(run.sessionId, branch.id, {
          runtimeLeafId: prepared.runtimeLeafId,
          runtimeSessionPath: prepared.runtimeSessionPath,
        });
        runtimeSessionPath = prepared.runtimeSessionPath ?? runtimeSessionPath;
      }

      await this.runtime.ensureSession({
        id: run.sessionId,
        title: session.title,
        branchId: run.branchId,
        sessionPath: runtimeSessionPath,
        recentSymbols: session.recentSymbols,
      });

      for await (const event of this.runtime.run({
        sessionId: run.sessionId,
        runId: run.id,
        content: run.input,
        branchId: run.branchId,
        sessionPath: runtimeSessionPath,
        workspaceContext,
        locale,
      })) {
        this.emit(event);
        if (event.type === 'message_delta' || event.type === 'message_completed') {
          answer = event.payload.answer;
        } else if (event.type === 'tool_completed') {
          toolCalls.push(event.payload.toolCall);
        } else if (event.type === 'run_failed') {
          failure = event.payload.error;
          sawTerminal = true;
        } else if (event.type === 'run_completed') {
          answer = event.payload.answer;
          sawTerminal = true;
        }
      }
    } catch (error) {
      failure = toApiError(error);
    }

    const active = this.activeRun;
    const cancelRequested = active?.cancelRequested ?? false;

    const now = this.now();
    const cancelled = Boolean(cancelRequested || (failure && failure.code === 'RUN_CANCELLED'));

    let runtimeArtifacts: RuntimeRunArtifacts | undefined;
    if (this.runtime.getRunArtifacts) {
      try {
        runtimeArtifacts = await this.runtime.getRunArtifacts({ sessionId: run.sessionId, runId: run.id });
      } catch {
        // Runtime diagnostics must never turn an already-settled generation
        // into a second failure.
      }
    }

    if (cancelled) {
      run.status = 'cancelled';
      run.answer = answer;
    } else if (failure) {
      run.status = 'failed';
      run.error = failure;
      run.answer = answer;
    } else {
      run.status = 'completed';
      run.answer = answer;
    }
    run.completedAt = now;
    if (runtimeArtifacts) {
      run.runtimeSessionId = runtimeArtifacts.runtimeSessionId;
      run.runtimeSessionPath = runtimeArtifacts.runtimeSessionPath;
      run.runtimeLeafId = runtimeArtifacts.runtimeLeafId;
      run.runtimeUserEntryId = runtimeArtifacts.runtimeUserEntryId;
      run.runtimeAssistantEntryId = runtimeArtifacts.runtimeAssistantEntryId;
      if (run.branchId) {
        await this.sessions.updateBranch(run.sessionId, run.branchId, {
          runtimeLeafId: runtimeArtifacts.runtimeLeafId,
          runtimeSessionPath: runtimeArtifacts.runtimeSessionPath,
        });
      }
    }
    if (run.manifest) {
      run.manifest = {
        ...run.manifest,
        toolCallIds: toolCalls.map((toolCall) => toolCall.id),
        runtimeSessionId: run.runtimeSessionId,
        runtimeSessionPath: run.runtimeSessionPath,
        runtimeLeafId: run.runtimeLeafId,
        runtimeUserEntryId: run.runtimeUserEntryId,
        runtimeAssistantEntryId: run.runtimeAssistantEntryId,
      };
    }

    await this.runs.update(run);

    // V8.1 §38–39: an *infrastructure* failure (Pi process failed to start /
    // stay up) is not an answer — do not persist an assistant-style message
    // that would spam the conversation. The renderer shows a dedicated banner
    // instead. Real failures (tool errors, task failures) keep the message.
    const isInfraFailure = run.status === 'failed' && isRuntimeInfraCode(run.error?.code);
    if (!isInfraFailure) {
      const assistantMessage: Message = {
        // The run id gives the assistant artifact a deterministic stable id
        // across the live event projection and a subsequent reload.
        id: `assistant-${run.id}`,
        role: 'assistant',
        content: answer || (run.status === 'failed' ? run.error?.message ?? 'Run failed.' : ''),
        timestamp: now,
        toolCalls: toolCalls.map(toRecord),
        branchId: run.branchId,
        parentMessageId: run.userMessageId,
        runId: run.id,
        generationId: run.generationId ?? run.id,
        operation: run.operation,
        sourceMessageId: run.sourceMessageId,
        contextSnapshot: run.contextSnapshot,
        runtimeEntryId: run.runtimeAssistantEntryId,
      };
      await this.sessions.appendMessage(run.sessionId, assistantMessage);
    }
    await this.sessions.updateSession(run.sessionId, {
      status: 'idle',
      recentSymbols: collectSymbols(toolCalls),
    });

    // The run is fully settled (persisted) only now; only then allow the next run.
    this.activeRun = null;

    // Adapters emit the terminal event themselves; synthesize it only when the
    // stream failed before producing one (e.g. runtime spawn failure), so the
    // UI always observes a terminal event.
    if (!sawTerminal) {
      if (cancelled) {
        this.emitRunEvent(run, 'run_failed', {
          error: { code: 'RUN_CANCELLED', message: 'Run cancelled by user.' },
        });
      } else if (failure) {
        this.emitRunEvent(run, 'run_failed', { error: failure });
      } else {
        this.emitRunEvent(run, 'run_completed', { answer, toolCalls });
      }
    }
  }

  private emitRunEvent(run: Run, type: AgentEvent['type'], payload?: AgentEventPayload): void {
    // Callers pair `type` with the matching payload shape.
    const event = {
      id: randomUUID(),
      sessionId: run.sessionId,
      runId: run.id,
      type,
      timestamp: this.now(),
      sequence: 1,
      payload,
    } as AgentEvent;
    this.emit(event);
  }

  private emit(event: AgentEvent): void {
    for (const listener of this.listeners) {
      listener(event);
    }
  }
}

function toRecord(toolCall: ToolCall): ToolCallRecord {
  return {
    id: toolCall.id,
    toolName: toolCall.toolName,
    args: toolCall.args,
    startedAt: toolCall.startedAt,
    completedAt: toolCall.completedAt,
    status: toolCall.status === 'error' ? 'error' : 'success',
    result: toolCall.result,
    error: toolCall.error,
  };
}

function collectSymbols(toolCalls: ToolCall[]): string[] {
  const symbols: string[] = [];
  for (const toolCall of toolCalls) {
    const symbol = typeof toolCall.args.symbol === 'string' ? toolCall.args.symbol.toUpperCase() : undefined;
    if (symbol && !symbols.includes(symbol)) {
      symbols.push(symbol);
    }
  }
  return symbols.slice(0, 5);
}

function findUserMessageIndex(messages: Message[], run: Run): number {
  if (run.userMessageId) {
    const exact = messages.findIndex((message) => message.id === run.userMessageId);
    if (exact >= 0) return exact;
  }
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index].role === 'user' && messages[index].content === run.input) return index;
  }
  return -1;
}

function findPreviousUserMessage(messages: Message[], beforeIndex: number): Message | undefined {
  for (let index = beforeIndex - 1; index >= 0; index -= 1) {
    if (messages[index].role === 'user') return messages[index];
  }
  return undefined;
}
