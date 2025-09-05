import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { indexDocument } from "./prepare.js";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { Pinecone } from "@pinecone-database/pinecone";
import { configDotenv } from "dotenv";
import { PineconeStore } from "@langchain/pinecone";
const filePath = "./ilovepdf_merged.pdf";
const doc = await indexDocument(filePath);
configDotenv();
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 100,
});
const chunkedDocs = await textSplitter.splitDocuments(doc);

const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GEMINI_API_KEY,
  model: "text-embedding-004",
});

const pinecone = new Pinecone();
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);


const store = await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
  pineconeIndex,
  maxConcurrency: 5,
});

console.log(store)

