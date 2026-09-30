import { OmniMemClient } from "../src/index.ts";

const client = new OmniMemClient({
  baseUrl: process.env.OMNIMEM_URL ?? "http://127.0.0.1:8765",
  apiKey: process.env.OMNIMEM_API_KEY,
  adminToken: process.env.OMNIMEM_ADMIN_TOKEN,
});

async function main(): Promise<void> {
  await client.memorize({
    content: "The team migrated the primary database to PostgreSQL 16 in Q2.",
    memory_type: "fact",
    privacy: "team",
  });

  const recall = await client.recall({ query: "what database do we use?", mode: "hybrid" });
  console.log("recalled:", recall.results ?? recall);

  const health = await client.health();
  console.log("health:", health.status);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
