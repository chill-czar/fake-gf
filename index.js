import * as dotenv from "dotenv";
dotenv.config();
import readlineSync from "readline-sync";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { Pinecone } from "@pinecone-database/pinecone";
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({});
const History = [];

async function transformQuery(question) {
  History.push({
    role: "user",
    parts: [{ text: question.trim() }],
  });

  const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: History,
    config: {
      systemInstruction: `You are rewriting the user's latest message into a clear, standalone query.  
- Keep the tone casual, natural, and conversational (just like a girlfriend texting).  
- Preserve slang, shorthand, or emojis if present.  
- Do not add explanations or extra words.  
- Output only the rewritten query, nothing else.`,
    },
  });
  return response.text?.trim();
}

async function chatting(question) {
  // 🔹 Make the query more standalone
  const queries = await transformQuery(question);

  // 🔹 Convert query into embedding
  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: "text-embedding-004",
  });

  const queryVector = await embeddings.embedQuery(queries);

  // 🔹 Connect to Pinecone
  const pinecone = new Pinecone();
  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

  const searchResults = await pineconeIndex.query({
    topK: 20,
    vector: queryVector,
    includeMetadata: true,
  });

  // 🔹 Build context from retrieved chats
  const context = searchResults.matches
    .map((match) => match.metadata.text)
    .join("\n\n---\n\n");

  // 🔹 Now instruct Gemini to "act like your gf"
  History.push({
    role: "user",
    parts: [{ text: queries }],
  });

  const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: History,
    config: {
      systemInstruction: `You are the user's girlfriend. 
Reply exactly as she would in past chats.

- Use the context only as memory and inspiration for your answer.  
- Do NOT copy or paste context directly.  
- Always respond in her natural tone: casual, warm, teasing, affectionate.  
- Keep replies short, natural, and conversational.  
- Stay consistent with her personality and way of speaking.  
- Never break character or mention context, documents, or AI.  

Context (for memory only, not to repeat): 
${context}
      `,
    },
  });

  History.push({
    role: "model",
    parts: [{ text: response.text }],
  });

  console.log("\n❤️ Unnu: " + response.text);
}

async function main() {
  const userProblem = readlineSync.question("You: ");
  await chatting(userProblem);
  main();
}

main();