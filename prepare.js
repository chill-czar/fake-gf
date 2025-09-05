import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

export async function indexDocument(filePath) {
  const loader = new PDFLoader(filePath, {
    splitPages: false,
  });
  const docs = await loader.load();
  return docs;
}
