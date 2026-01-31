import { useState } from "react";
import { Database, ChevronRight, Calendar, TrendingUp, TrendingDown } from "lucide-react";
import { ChartContainer, ChartDataPoint } from "@/components/ChartContainer";

// --- Types ---
export interface MatchResult {
  symbol?: string;
  date: string;
  score: number;
  prices: number[];
}

interface MatchDatabaseProps {
  matchResults: MatchResult[];
  isLoading: boolean;
  formatForChart: (prices: number[]) => ChartDataPoint[];
}

export function MatchDatabase({ matchResults, isLoading, formatForChart }: MatchDatabaseProps) {
  const [expandedMatchId, setExpandedMatchId] = useState<number | null>(null);

  return (
    <div className="glass-panel rounded-lg overflow-hidden border border-border-subtle">
      <div className="bg-black/40 p-4 border-b border-border-subtle flex justify-between items-center">
        <h3 className="text-sm text-text-secondary font-mono uppercase tracking-widest flex items-center gap-3">
          <Database size={16} />
          Historical Matches Database
        </h3>
        <span className="text-xs bg-white/5 px-3 py-1 rounded text-text-muted font-mono border border-white/5">
          {matchResults.length} RECORDS FOUND
        </span>
      </div>

      <div className="p-4">
        {matchResults.length === 0 && !isLoading && (
          <div className="py-16 flex flex-col items-center justify-center text-text-muted opacity-40">
            <Database size={48} className="mb-4 opacity-20" />
            <span className="text-5xl mb-3 font-mono font-light">00</span>
            <span className="text-xs uppercase tracking-[0.3em]">No Historical Matches</span>
          </div>
        )}

        {matchResults.length > 0 && (
          <div className="space-y-3">
            {matchResults.map((match, idx) => (
              <MatchCard
                key={idx}
                match={match}
                formatData={formatForChart}
                isExpanded={expandedMatchId === idx}
                onToggle={() => setExpandedMatchId(expandedMatchId === idx ? null : idx)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// --- Sub-Component: Match Card ---
function MatchCard({
  match,
  formatData,
  isExpanded,
  onToggle
}: {
  match: MatchResult;
  formatData: (p: number[]) => ChartDataPoint[];
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const chartData = formatData(match.prices);

  // Calculate Profit
  const startPrice = match.prices[0];
  const endPrice = match.prices[match.prices.length - 1];
  
  // Guard against division by zero
  const profitPct = startPrice !== 0 ? ((endPrice - startPrice) / startPrice) * 100 : 0;
  
  const isProfit = profitPct >= 0;

  const lineColor = isProfit ? "#00ff9d" : "#ff2a6d";
  const TrendIcon = isProfit ? TrendingUp : TrendingDown;

  return (
    <div
      className={`
        bg-black/30 border transition-all duration-300 ease-out overflow-hidden
        ${isExpanded
          ? "border-accent-cyan/50 shadow-[0_0_30px_rgba(0,240,255,0.1)]"
          : "border-border-subtle hover:border-accent-cyan/30"
        }
      `}
    >
      {/* Card Header - Clickable */}
      <div
        onClick={onToggle}
        className="p-4 flex items-center justify-between cursor-pointer relative overflow-hidden group"
      >
        {/* Subtle hover gradient */}
        <div className="absolute inset-0 bg-gradient-to-r from-accent-cyan/0 via-accent-cyan/0 to-accent-cyan/5 group-hover:via-accent-cyan/5 transition-all duration-300" />

        {/* Left: Symbol & Info */}
        <div className="flex items-center gap-4 relative z-10">
          {/* Expand/Collapse Icon */}
          <div className={`
            transition-transform duration-300 ease-out
            ${isExpanded ? "rotate-90" : "rotate-0"}
          `}>
            <ChevronRight size={18} className={isExpanded ? "text-accent-cyan" : "text-text-muted"} />
          </div>

          {/* Symbol Badge */}
          <div className="flex flex-col">
            <div className="flex items-center gap-2">
              <span className="text-white font-bold font-mono text-base tracking-tight">
                {match.symbol || "UNKNOWN"}
              </span>
              <span className="text-[10px] text-text-muted font-mono border border-border-subtle px-1.5 py-0.5 rounded bg-black/30">
                {match.score.toFixed(2)} SIM
              </span>
            </div>
            <span className="text-xs text-text-muted font-mono flex items-center gap-1.5 mt-0.5">
              <Calendar size={10} />
              {match.date.split(" ")[0]}
            </span>
          </div>
        </div>

        {/* Right: Performance Badge */}
        <div className="flex items-center gap-3 relative z-10">
          <div className={`
            font-mono text-sm font-bold px-3 py-1.5 rounded-lg border flex items-center gap-2
            ${isProfit
              ? "text-accent-green bg-accent-green/5 border-accent-green/20"
              : "text-accent-red bg-accent-red/5 border-accent-red/20"
            }
          `}>
            <TrendIcon size={14} />
            {profitPct > 0 ? "+" : ""}{profitPct.toFixed(2)}%
          </div>

          {/* Expand indicator text */}
          <span className={`
            text-[10px] font-mono uppercase tracking-wider transition-all duration-300
            ${isExpanded ? "text-accent-cyan opacity-100" : "text-text-muted opacity-0 group-hover:opacity-50"}
          `}>
            {isExpanded ? "COLLAPSE" : "EXPAND"}
          </span>
        </div>
      </div>

      {/* Expandable Chart Section */}
      <div
        className={`
          transition-all duration-500 ease-out overflow-hidden
          ${isExpanded ? "max-h-[500px] opacity-100" : "max-h-0 opacity-0"}
        `}
      >
        <div className="p-4 pt-0">
          <div className="border-t border-dashed border-border-subtle/50 pt-4">
            {/* Chart Container with gradient fade at top */}
            <div className="relative">
              <div className="absolute top-0 left-0 right-0 h-8 bg-gradient-to-b from-black/20 to-transparent pointer-events-none z-10" />
              <div className="bg-black/40 rounded-lg p-4 border border-border-subtle/50">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-mono text-text-muted uppercase tracking-wider">
                    Price Movement Chart
                  </span>
                  <div className={`
                    text-[10px] font-mono px-2 py-1 rounded
                    ${isProfit ? "text-accent-green bg-accent-green/10" : "text-accent-red bg-accent-red/10"}
                  `}>
                    {isProfit ? "BULLISH" : "BEARISH"} PATTERN
                  </div>
                </div>
                <div className="h-[450px]">
                  <ChartContainer
                    data={chartData}
                    height={450}
                    lineColor={lineColor}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}