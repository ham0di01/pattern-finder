import { Activity } from "lucide-react";

export function Header() {
  return (
    <header className="flex justify-between items-end border-b border-border-subtle pb-4">
      <div>
        <div className="flex items-center gap-2 mb-1">
          <Activity className="text-accent-cyan" size={20} />
          <h1 className="text-2xl md:text-3xl font-bold tracking-tighter text-white uppercase font-mono">
            Pattern<span className="text-accent-cyan">Trader</span>
          </h1>
          <span className="text-[10px] bg-accent-cyan/10 text-accent-cyan px-2 py-0.5 rounded border border-accent-cyan/20">
            v2.0.4
          </span>
        </div>
        <p className="text-text-secondary text-xs font-mono uppercase tracking-widest">
          Algorithmic Pattern Recognition System
        </p>
      </div>
      <div className="text-right hidden md:block">
        <div className="text-xs text-text-muted font-mono">SYSTEM STATUS</div>
        <div className="flex items-center justify-end gap-2 text-accent-green text-sm font-bold">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent-green opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-accent-green"></span>
          </span>
          ONLINE
        </div>
      </div>
    </header>
  );
}
