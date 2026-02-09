import { Client, Thread, ThreadState } from '@langchain/langgraph-sdk';

export class LangGraphBase {
  client: Client;

  constructor(client: Client) {
    this.client = client;
  }

  /**
   * Creates a new thread with optional metadata
   */
  async createThread(metadata?: Record<string, unknown>) {
    return this.client.threads.create({ metadata });
  }

  /**
   * Gets a thread by ID
   */
  async getThread(threadId: string): Promise<Thread> {
    return this.client.threads.get(threadId);
  }

  /**
   * Searches for threads with optional metadata filters
   */
  async searchThreads(params: {
    metadata?: Record<string, unknown>;
    limit?: number;
    offset?: number;
  }): Promise<Thread[]> {
    return this.client.threads.search({
      metadata: params.metadata,
      limit: params.limit || 10,
      offset: params.offset || 0,
    });
  }

  /**
   * Gets the current state of a thread
   */
  async getThreadState<
    T extends Record<string, unknown> = Record<string, unknown>,
  >(
    threadId: string,
  ): Promise<ThreadState<T>> {
    return this.client.threads.getState(threadId);
  }

  /**
   * Updates the state of a thread
   */
  async updateThreadState(
    threadId: string,
    values: Record<string, unknown>,
    asNode?: string,
  ) {
    return this.client.threads.updateState(threadId, {
      values,
      asNode,
    });
  }

  /**
   * Deletes a thread by ID
   */
  async deleteThread(threadId: string) {
    return this.client.threads.delete(threadId);
  }

  /**
   * Gets the history of a thread's states
   */
  async getThreadHistory(threadId: string, limit = 10) {
    return this.client.threads.getHistory(threadId, { limit });
  }

  /**
   * Utility function to check if a thread is interrupted
   */
  isThreadInterrupted(thread: Thread): boolean {
    return !!(thread.interrupts && Object.keys(thread.interrupts).length > 0);
  }

  /**
   * Utility function to get interrupts from a thread
   */
  getThreadInterrupts(thread: Thread): unknown[] | undefined {
    if (!thread.interrupts) return undefined;

    return Object.values(thread.interrupts).flatMap((interrupt) => {
      if (Array.isArray(interrupt)) {
        const first = interrupt[0];
        if (Array.isArray(first)) {
          const candidate = first[1] as { value?: unknown } | undefined;
          return candidate?.value !== undefined ? [candidate.value] : [];
        }

        return interrupt
          .map((item) => (item as { value?: unknown }).value)
          .filter((value): value is unknown => value !== undefined);
      }

      const value = (interrupt as { value?: unknown }).value;
      return value !== undefined ? [value] : [];
    });
  }

  /**
   * Utility function to resume an interrupted thread
   */
  async resumeThread(
    threadId: string,
    assistantId: string,
    resumeValue: unknown,
    config?: {
      configurable?: { [key: string]: unknown };
    },
  ) {
    return this.client.runs.stream(threadId, assistantId, {
      command: { resume: resumeValue },
      config: {
        configurable: config?.configurable,
      },
    });
  }
}
