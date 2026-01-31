import { useRef } from "react";
import { UploadCloud, Play, ImageIcon, ScanLine } from "lucide-react";

interface UploadTerminalProps {
  selectedImage: File | null;
  previewUrl: string | null;
  debugImage: string | null;
  isLoading: boolean;
  availableSymbols: string[];
  selectedSymbols: Set<string>;
  isGlobalMode: boolean;
  onImageSelected: (file: File) => void;
  onModeChange: (isGlobal: boolean) => void;
  onSymbolToggle: (symbol: string, isSelected: boolean) => void;
  onAnalyze: () => void;
}

export function UploadTerminal({
  selectedImage,
  previewUrl,
  debugImage,
  isLoading,
  availableSymbols,
  selectedSymbols,
  isGlobalMode,
  onImageSelected,
  onModeChange,
  onSymbolToggle,
  onAnalyze,
}: UploadTerminalProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onImageSelected(file);
    }
  };

  return (
    <div className="lg:col-span-3 flex flex-col">
      <div className="glass-panel rounded-lg p-1 flex-1 flex flex-col tech-border">
        <div className="bg-black/40 p-2 border-b border-border-subtle flex justify-between items-center">
          <span className="text-[10px] text-accent-cyan font-mono uppercase tracking-wider">
            01 // Input Source
          </span>
          <ImageIcon size={14} className="text-text-muted" />
        </div>

        <div className="p-4 flex-1 flex flex-col gap-4">
          <div
            className="flex-1 min-h-[280px] border border-dashed border-border-subtle hover:border-accent-cyan/50 transition-colors rounded bg-black/20 relative overflow-hidden group cursor-pointer"
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              accept="image/*"
              className="hidden"
              ref={fileInputRef}
              onChange={handleFileChange}
            />

            {/* Grid Overlay */}
            <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:20px_20px] pointer-events-none" />

            {previewUrl ? (
              <>
                <img
                  src={previewUrl}
                  alt="Selected chart for analysis"
                  className="absolute inset-0 w-full h-full object-cover opacity-100"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent flex items-end justify-center pb-4 opacity-0 group-hover:opacity-100 transition-opacity">
                  <span className="text-xs font-mono text-accent-cyan bg-black/80 px-2 py-1 border border-accent-cyan/30 rounded">
                    CHANGE_SOURCE
                  </span>
                </div>
              </>
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-text-muted gap-3">
                <div className="p-4 rounded-full bg-accent-cyan/5 border border-accent-cyan/20 group-hover:bg-accent-cyan/10 transition-colors">
                  <UploadCloud size={24} className="text-accent-cyan" />
                </div>
                <div className="text-center">
                  <p className="text-sm font-bold text-text-secondary group-hover:text-accent-cyan transition-colors">
                    UPLOAD CHART
                  </p>
                  <p className="text-[10px] font-mono mt-1 opacity-50">
                    SUPPORTS JPG, PNG, WEBP
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="space-y-4">
            {/* Coin Selection Panel */}
            <div className="glass-panel rounded-lg p-3 border border-border-subtle">
              <div className="flex items-center justify-between mb-3">
                <span className="text-[10px] text-accent-cyan font-mono uppercase tracking-wider">
                  02 // Search Scope
                </span>
                <span className="text-[9px] text-text-muted font-mono">
                  {availableSymbols.length} SYMBOLS
                </span>
              </div>

              {/* Mode Toggle */}
              <div className="flex gap-2 mb-3">
                <button
                  onClick={() => onModeChange(true)}
                  className={`flex-1 py-2 px-3 rounded text-[10px] font-mono font-bold uppercase transition-all ${
                    isGlobalMode
                      ? "bg-accent-cyan/20 border-accent-cyan text-accent-cyan"
                      : "bg-black/30 border-border-subtle text-text-muted hover:border-accent-cyan/30"
                  } border`}
                >
                  Global
                </button>
                <button
                  onClick={() => onModeChange(false)}
                  className={`flex-1 py-2 px-3 rounded text-[10px] font-mono font-bold uppercase transition-all ${
                    !isGlobalMode
                      ? "bg-accent-cyan/20 border-accent-cyan text-accent-cyan"
                      : "bg-black/30 border-border-subtle text-text-muted hover:border-accent-cyan/30"
                  } border`}
                >
                  Select Coins
                </button>
              </div>

              {/* Symbol Multi-Select (shown only in Select mode) */}
              {!isGlobalMode && (
                <div className="max-h-40 overflow-y-auto bg-black/30 rounded border border-border-subtle p-2 space-y-1">
                  {availableSymbols.length === 0 ? (
                    <div className="text-center py-4">
                      <span className="text-[10px] text-text-muted font-mono">
                        Loading symbols...
                      </span>
                    </div>
                  ) : (
                    availableSymbols.map((symbol) => (
                      <label
                        key={symbol}
                        className="flex items-center gap-2 p-1.5 rounded hover:bg-white/5 cursor-pointer transition-colors"
                      >
                        <input
                          type="checkbox"
                          checked={selectedSymbols.has(symbol)}
                          onChange={(e) =>
                            onSymbolToggle(symbol, e.target.checked)
                          }
                          className="w-3 h-3 rounded border-border-subtle bg-black/50 accent-accent-cyan"
                        />
                        <span className="text-[10px] font-mono text-text-secondary">
                          {symbol}
                        </span>
                      </label>
                    ))
                  )}
                </div>
              )}

              {/* Selection Summary */}
              <div className="mt-2 flex justify-between items-center text-[9px] font-mono text-text-muted">
                <span>
                  MODE: {isGlobalMode ? "GLOBAL SEARCH" : "SPECIFIC COINS"}
                </span>
                {!isGlobalMode && <span>{selectedSymbols.size} SELECTED</span>}
              </div>
            </div>
            <button
              onClick={onAnalyze}
              disabled={
                !selectedImage ||
                isLoading ||
                (!isGlobalMode && selectedSymbols.size === 0)
              }
              className="w-full py-3 bg-accent-cyan/10 hover:bg-accent-cyan/20 border border-accent-cyan/50 hover:border-accent-cyan text-accent-cyan disabled:opacity-30 disabled:hover:bg-accent-cyan/10 disabled:hover:border-accent-cyan/50 rounded font-mono font-bold text-sm uppercase tracking-wider transition-all flex items-center justify-center gap-2 group"
            >
              {isLoading ? (
                <>
                  <span className="animate-spin h-3 w-3 border-2 border-current border-t-transparent rounded-full" />
                  SCANNING...
                </>
              ) : (
                <>
                  EXECUTE_SCAN
                  <Play size={14} className="group-hover:fill-current" />
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Debug View */}
      {debugImage && (
        <div className="glass-panel rounded-lg p-1 flex flex-col h-48 mt-6">
          <div className="bg-black/40 p-2 border-b border-border-subtle flex justify-between items-center">
            <span className="text-[10px] text-text-muted font-mono uppercase">
              VISION_DEBUG
            </span>
            <ScanLine size={12} className="text-text-muted" />
          </div>
          <div className="flex-1 p-2 overflow-hidden relative">
            <img
              src={`data:image/png;base64,${debugImage}`}
              alt="Processed debug view"
              className="w-full h-full object-contain opacity-60 grayscale hover:grayscale-0 transition-all"
            />
            <div className="absolute top-2 right-2 text-[8px] font-mono text-accent-green bg-black/80 px-1 border border-accent-green/30">
              PROCESSED
            </div>
          </div>
        </div>
      )}
    </div>
  );
}