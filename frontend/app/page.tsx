"use client";

import { useState, useEffect } from "react";
import { ScanLine } from "lucide-react";

import { Header } from "@/components/Header";
import { UploadTerminal } from "@/components/UploadTerminal";
import { AnalysisDashboard } from "@/components/AnalysisDashboard";
import { MatchDatabase, MatchResult } from "@/components/MatchDatabase";
import { ChartDataPoint } from "@/components/ChartContainer";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const [extractedPattern, setExtractedPattern] = useState<number[]>([]);
  const [matchResults, setMatchResults] = useState<MatchResult[]>([]);
  const [debugImage, setDebugImage] = useState<string | null>(null);

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [availableSymbols, setAvailableSymbols] = useState<string[]>([]);
  const [selectedSymbols, setSelectedSymbols] = useState<Set<string>>(new Set());
  const [isGlobalMode, setIsGlobalMode] = useState(true);

  // Fetch available symbols on component mount
  useEffect(() => {
    const fetchSymbols = async () => {
      try {
        const res = await fetch(`${API_URL}/api/v1/symbols`);
        if (!res.ok) throw new Error("Failed to fetch symbols");
        const data = await res.json();
        setAvailableSymbols(data.symbols || []);
      } catch (err) {
        console.error("Failed to load symbols:", err);
        // Silent error for symbols, or show a toast
      }
    };
    fetchSymbols();
  }, []);

  // Clean up object URL to prevent memory leaks
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  // --- Normalization Logic (Percentage Change) ---
  const formatForChart = (prices: number[]): ChartDataPoint[] => {
    if (!prices || prices.length === 0) return [];
    const startValue = prices[0];
    const now = new Date();
    
    // Avoid division by zero if startValue is 0 (though unlikely for price data)
    const safeStartValue = startValue === 0 ? 1 : Math.abs(startValue);

    return prices.map((value, i) => {
      // (Current - Start) / Start * 100
      const pctChange = ((value - startValue) / safeStartValue) * 100;
      
      const date = new Date(now.getTime() - (prices.length - 1 - i) * 60 * 60 * 1000);
      return {
        time: Math.floor(date.getTime() / 1000).toString(),
        value: pctChange,
      };
    });
  };

  const handleImageUpload = (file: File) => {
    // Revoke previous URL if it exists
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    
    setSelectedImage(file);
    setPreviewUrl(URL.createObjectURL(file));
    
    // Reset state
    setMatchResults([]);
    setExtractedPattern([]);
    setDebugImage(null);
    setError(null);
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;
    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedImage);

    // Send mode based on selection
    if (isGlobalMode) {
      formData.append("mode", "GLOBAL");
    } else {
      formData.append("mode", Array.from(selectedSymbols).join(','));
    }

    try {
      const response = await fetch(`${API_URL}/api/v1/analyze-image`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API Error: ${response.status} ${errorText}`);
      }

      const data = await response.json();
      
      // Validate response structure
      if (!data.extracted_pattern || !data.matches) {
        throw new Error("Invalid response format from server");
      }

      setExtractedPattern(data.extracted_pattern);
      setMatchResults(data.matches);
      setDebugImage(data.debug_image || null);
    } catch (err: unknown) {
      console.error("Analysis failed:", err);
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Failed to analyze image. An unknown error occurred.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleSymbolToggle = (symbol: string, isSelected: boolean) => {
    const newSet = new Set(selectedSymbols);
    if (isSelected) {
      newSet.add(symbol);
    } else {
      newSet.delete(symbol);
    }
    setSelectedSymbols(newSet);
  };

  const handleModeChange = (global: boolean) => {
      setIsGlobalMode(global);
      if (global) {
          setSelectedSymbols(new Set());
      }
  };

  // Prepare Main Chart Data
  const mainChartData = formatForChart(extractedPattern);

  return (
    <main className="min-h-screen p-4 md:p-8 font-sans selection:bg-cyan-500/30">
      <div className="max-w-[1600px] mx-auto space-y-6">
        
        <Header />

        {/* ERROR */}
        {error && (
          <div className="bg-red-950/30 border border-red-500/50 text-red-200 p-4 rounded backdrop-blur-sm flex items-center gap-3">
            <ScanLine className="text-red-500" />
            <span className="font-mono text-sm">{error}</span>
          </div>
        )}

        <div className="space-y-6">

          {/* TOP SECTION: Input Terminal + Pattern Recognition */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

            <UploadTerminal 
              selectedImage={selectedImage}
              previewUrl={previewUrl}
              debugImage={debugImage}
              isLoading={isLoading}
              availableSymbols={availableSymbols}
              selectedSymbols={selectedSymbols}
              isGlobalMode={isGlobalMode}
              onImageSelected={handleImageUpload}
              onModeChange={handleModeChange}
              onSymbolToggle={handleSymbolToggle}
              onAnalyze={analyzeImage}
            />

            <div className="lg:col-span-9 flex flex-col">
              <AnalysisDashboard 
                extractedPattern={extractedPattern}
                mainChartData={mainChartData}
              />
            </div>
          </div>

          {/* BOTTOM SECTION: Historical Matches Database */}
          <MatchDatabase 
            matchResults={matchResults}
            isLoading={isLoading}
            formatForChart={formatForChart}
          />
        </div>
      </div>
    </main>
  );
}
