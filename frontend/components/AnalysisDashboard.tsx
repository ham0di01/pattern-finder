import { Terminal, Activity } from "lucide-react";
import { ChartContainer, ChartDataPoint } from "@/components/ChartContainer";

interface AnalysisDashboardProps {
  extractedPattern: number[];
  mainChartData: ChartDataPoint[];
}

export function AnalysisDashboard({ extractedPattern, mainChartData }: AnalysisDashboardProps) {
  return (
    <div className="glass-panel rounded-lg flex-1 flex flex-col tech-border relative overflow-hidden">
      {/* Decorative corner lines */}
      <div className="absolute top-0 right-0 p-2">
        <div className="w-16 h-[1px] bg-accent-cyan/20"></div>
      </div>

      <div className="bg-black/40 p-3 border-b border-border-subtle flex justify-between items-center">
        <h3 className="text-xs text-accent-cyan font-mono uppercase tracking-widest flex items-center gap-2">
          <Terminal size={14} /> Pattern Recognition Module
        </h3>
        <div className="flex gap-2">
          <div className="w-2 h-2 rounded-full bg-text-muted/30"></div>
          <div className="w-2 h-2 rounded-full bg-text-muted/30"></div>
          <div className="w-2 h-2 rounded-full bg-accent-cyan/50"></div>
        </div>
      </div>

      <div className="flex-1 p-4 flex flex-col relative">
        {extractedPattern.length > 0 ? (
          <>
            <div className="flex justify-between items-center mb-4 font-mono text-xs">
              <span className="text-text-muted">
                DETECTED_PATTERN_LENGTH: <span className="text-white">{extractedPattern.length} pts</span>
              </span>
              <span className="text-accent-green animate-pulse">● LIVE</span>
            </div>
            <div className="flex-1 w-full h-full min-h-[400px]">
              <ChartContainer
                data={mainChartData}
                height={450}
                lineColor="#00f0ff"
              />
            </div>
          </>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-text-muted gap-4 border border-dashed border-border-subtle/30 rounded m-4 bg-black/20">
            <div className="w-16 h-16 rounded-full border border-border-subtle flex items-center justify-center">
              <Activity className="opacity-20" size={32} />
            </div>
            <div className="text-center font-mono">
              <p className="text-sm">AWAITING INPUT STREAM</p>
              <p className="text-xs opacity-50 mt-1">Upload chart to initialize analysis</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}