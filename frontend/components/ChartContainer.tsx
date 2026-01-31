"use client";

import { useEffect, useRef, useState } from "react";
import { createChart, ColorType, LineSeries, UTCTimestamp, IChartApi, ISeriesApi, MouseEventParams } from "lightweight-charts";

export interface ChartDataPoint {
  time: string | number;
  value: number;
}

interface ChartProps {
  data: ChartDataPoint[];
  matches?: ChartDataPoint[][]; 
  height?: number; 
  lineColor?: string; 
}

export const ChartContainer = ({ 
  data, 
  matches, 
  height = 400, 
  lineColor = "#00f0ff" 
}: ChartProps) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const mainSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const matchSeriesRefs = useRef<ISeriesApi<"Line">[]>([]);
  
  const [showMatches, setShowMatches] = useState(false);
  const showMatchesRef = useRef(showMatches);

  // Keep ref in sync with state for access inside the creation effect without adding dependency
  useEffect(() => {
    showMatchesRef.current = showMatches;
  }, [showMatches]);

  // Update visibility of existing series when state changes
  useEffect(() => {
    matchSeriesRefs.current.forEach((series) => {
      series.applyOptions({ visible: showMatches });
    });
  }, [showMatches]);

  // Helper to ensure time is a number if it's a timestamp string
  const formatData = (d: ChartDataPoint[]) => d.map(item => ({
    ...item,
    time: (typeof item.time === 'string' && !isNaN(Number(item.time))) 
          ? Number(item.time) as UTCTimestamp 
          : item.time
  })).sort((a, b) => (a.time as number) - (b.time as number));

  // Initialize Chart (Runs only on mount or height change)
  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#525260", // Muted text
      },
      grid: {
        vertLines: { color: "#1f1f26" }, // Subtle border
        horzLines: { color: "#1f1f26" },
      },
      width: chartContainerRef.current.clientWidth,
      height: height,
      timeScale: {
        visible: true,
        borderVisible: false,
        timeVisible: true,
      },
      rightPriceScale: {
        borderVisible: false,
      },
    });

    chartRef.current = chart;

    // Create Main Series
    const mainSeries = chart.addSeries(LineSeries, {
      color: lineColor,
      lineWidth: 2,
      priceFormat: {
        type: 'percent',
        precision: 2,
      },
    });
    mainSeriesRef.current = mainSeries;

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    const handleClick = (param: MouseEventParams) => {
      if (param.point) {
        setShowMatches((prev) => !prev);
      }
    };

    window.addEventListener("resize", handleResize);
    chart.subscribeClick(handleClick);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
      chartRef.current = null;
      mainSeriesRef.current = null;
      matchSeriesRefs.current = [];
    };
  }, [height]); // Re-init only if height changes (lineColor is handled in update effect)

  // Update Data and Options
  useEffect(() => {
    if (!chartRef.current || !mainSeriesRef.current) return;

    // Update Main Series
    mainSeriesRef.current.applyOptions({ color: lineColor });
    
    if (data && data.length > 0) {
      mainSeriesRef.current.setData(formatData(data));
    } else {
      mainSeriesRef.current.setData([]);
    }

    // Update Matches (Clear old ones and add new ones)
    matchSeriesRefs.current.forEach(series => {
      if (chartRef.current) {
        chartRef.current.removeSeries(series);
      }
    });
    matchSeriesRefs.current = [];

    // Add new match series
    if (matches && matches.length > 0) {
      matches.forEach((matchData, index) => {
        if (!chartRef.current) return;
        
        const line = chartRef.current.addSeries(LineSeries, {
          color: `rgba(255, 255, 255, ${0.3 - index * 0.05})`,
          lineWidth: 1,
          lineStyle: 2,
          visible: showMatchesRef.current,
        });
        
        line.setData(formatData(matchData));
        matchSeriesRefs.current.push(line);
      });
    }

    chartRef.current.timeScale().fitContent();

  }, [data, matches, lineColor]); 

  return <div ref={chartContainerRef} className="w-full overflow-hidden rounded-lg cursor-pointer" />;
};
