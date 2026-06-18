import { DemoWalkthrough } from "./components/DemoWalkthrough";
import { Header } from "./components/Header";
import { Hero } from "./components/Hero";

export default function Home() {
  return (
    <main className="min-h-[100dvh] overflow-x-hidden bg-[var(--paper)]">
      <Header />
      <Hero />
      <DemoWalkthrough />
    </main>
  );
}
