/**
 * llm.remote.ts - Remote LLM implementation for QMD
 * Supports OpenAI-compatible APIs and Google Gemini API (via Proxy)
 */

import { 
  type LLM, 
  type EmbeddingResult, 
  type GenerateResult, 
  type ModelInfo, 
  type Queryable, 
  type RerankResult,
  type EmbedOptions,
  type GenerateOptions,
  type RerankOptions,
  type RerankDocument,
  formatQueryForEmbedding,
  formatDocForEmbedding
} from "./llm.js";

export class RemoteLLM implements LLM {
  private gatewayUrl: string;
  private embedModel: string;
  private generateModel: string;

  constructor() {
    // Default to our new smart gateway
    this.gatewayUrl = process.env.QMD_REMOTE_URL || "https://gemini-gateway.mukhtharcm.workers.dev/v1";
    this.embedModel = process.env.QMD_REMOTE_EMBED_MODEL || "gemini-embedding-001";
    this.generateModel = process.env.QMD_REMOTE_GENERATE_MODEL || "gemini-2.0-flash-exp";
  }

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    const formattedText = options.isQuery ? formatQueryForEmbedding(text) : formatDocForEmbedding(text, options.title);
    const model = this.embedModel;

    try {
      const response = await fetch(`${this.gatewayUrl}/embeddings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input: formattedText, model: model })
      });

      if (!response.ok) {
        const err = await response.text();
        throw new Error(`Gateway embedding failed (${response.status}): ${err}`);
      }

      const data = await response.json() as any;
      const rawVector = data.data[0].embedding;
      
      // Slicing to 768 to ensure compatibility with QMD's internal SQLite-vec schema.
      const slicedVector = rawVector.slice(0, 768);

      return {
        embedding: slicedVector,
        model: model
      };
    } catch (error) {
      console.error("Remote embedding error:", error);
      return null;
    }
  }

  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerateResult | null> {
    const model = this.generateModel;
    try {
      const response = await fetch(`${this.gatewayUrl}/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [{ role: "user", content: prompt }],
          model: model,
          max_tokens: options.maxTokens || 1000,
          temperature: options.temperature ?? 1
        })
      });

      if (!response.ok) {
        const err = await response.text();
        throw new Error(`Gateway generation failed (${response.status}): ${err}`);
      }

      const data = await response.json() as any;
      return {
        text: data.choices[0].message.content,
        model: model,
        done: true
      };
    } catch (error) {
      console.error("Remote generation error:", error);
      return null;
    }
  }

  async modelExists(model: string): Promise<ModelInfo> { return { name: model, exists: true }; }
  
  async tokenize(text: string): Promise<any[]> {
    return text.split(/\s+/).filter(Boolean);
  }

  async countTokens(text: string): Promise<number> {
    const tokens = await this.tokenize(text);
    return tokens.length;
  }

  async detokenize(tokens: any[]): Promise<string> {
    return tokens.join(" ");
  }

  async expandQuery(query: string, options: { context?: string; includeLexical?: boolean } = {}): Promise<Queryable[]> {
    const prompt = `You are a search expert. Output exactly 1-3 'lex' lines, 1-3 'vec' lines, and MAX ONE 'hyde' line.

Query: ${query}

Format:
lex: keyword
vec: semantic
hyde: hypothetical passage

Final Output:`;

    const res = await this.generate(prompt);
    if (!res) return [{ type: 'vec', text: query }];

    const lines = res.text.trim().split("\n");
    const queryables: Queryable[] = lines.map(line => {
      const colonIdx = line.indexOf(":");
      if (colonIdx === -1) return null;
      const type = line.slice(0, colonIdx).trim() as any;
      const text = line.slice(colonIdx + 1).trim();
      if (type !== 'lex' && type !== 'vec' && type !== 'hyde') return null;
      return { type: type as any, text };
    }).filter((q): q is Queryable => q !== null);

    const includeLex = options.includeLexical ?? true;
    return includeLex ? queryables : queryables.filter(q => q.type !== 'lex');
  }

  async rerank(query: string, documents: RerankDocument[], options: RerankOptions = {}): Promise<RerankResult> {
    return {
      results: documents.map((doc, index) => ({ file: doc.file, score: 1.0 - (index / documents.length), index })),
      model: "passthrough"
    };
  }

  async dispose(): Promise<void> {}
}

export default RemoteLLM;
