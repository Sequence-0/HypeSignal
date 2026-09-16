import React from 'react';
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';
import { NuancedEmotions } from '../../types';

interface EmotionRadarProps {
  emotions: NuancedEmotions;
}

export const EmotionRadar: React.FC<EmotionRadarProps> = ({ emotions }) => {
  const chartData = [
    { emotion: 'Joy', value: Math.round((emotions.joy || 0) * 100), fullMark: 100 },
    { emotion: 'Optimism', value: Math.round((emotions.optimism || 0) * 100), fullMark: 100 },
    { emotion: 'Excitement', value: Math.round((emotions.excitement || 0) * 100), fullMark: 100 },
    { emotion: 'Surprise', value: Math.round((emotions.surprise || 0) * 100), fullMark: 100 },
    { emotion: 'Anxiety', value: Math.round((emotions.anxiety || 0) * 100), fullMark: 100 },
    { emotion: 'Sadness', value: Math.round((emotions.sadness || 0) * 100), fullMark: 100 },
    { emotion: 'Anger', value: Math.round((emotions.anger || 0) * 100), fullMark: 100 },
    { emotion: 'Disgust', value: Math.round((emotions.disgust || 0) * 100), fullMark: 100 },
    { emotion: 'Fear', value: Math.round((emotions.fear || 0) * 100), fullMark: 100 },
  ];

  const sorted = [...chartData].sort((a, b) => b.value - a.value);

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-bold text-slate-900">9-Dimensional Emotion Radar</h3>
          <p className="text-xs text-slate-500">Fine-grained psychological and emotional frequencies</p>
        </div>
        <span className="text-xs font-semibold px-2.5 py-1 bg-indigo-50 text-indigo-700 border border-indigo-200 rounded-full">
          Cardiff RoBERTa Calibrated
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
        <div className="md:col-span-2 h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart cx="50%" cy="50%" outerRadius="75%" data={chartData}>
              <PolarGrid stroke="#e2e8f0" />
              <PolarAngleAxis
                dataKey="emotion"
                tick={{ fill: '#475569', fontSize: 11, fontWeight: 500 }}
              />
              <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#cbd5e1" tick={{ fontSize: 10 }} />
              <Radar
                name="Frequency (%)"
                dataKey="value"
                stroke="#6366f1"
                fill="#6366f1"
                fillOpacity={0.3}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  borderRadius: '8px',
                  color: '#fff',
                  border: 'none',
                  fontSize: '12px',
                }}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Emotion breakdown list */}
        <div className="space-y-2">
          <div className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">
            Top Emotional Drivers
          </div>
          {sorted.slice(0, 5).map((item, idx) => (
            <div key={item.emotion} className="flex items-center justify-between text-xs py-1 border-b border-slate-100 last:border-0">
              <span className="font-medium text-slate-700 flex items-center gap-1.5">
                <span className="w-4 text-slate-400 font-mono text-[10px]">#{idx + 1}</span>
                {item.emotion}
              </span>
              <div className="flex items-center gap-2">
                <div className="w-16 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-indigo-600 rounded-full"
                    style={{ width: `${item.value}%` }}
                  />
                </div>
                <span className="font-bold text-slate-800 w-8 text-right">{item.value}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
