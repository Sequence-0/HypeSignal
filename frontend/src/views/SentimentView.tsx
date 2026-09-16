import React, { useState } from 'react';
import {
  Smile,
  Zap,
  AlertCircle,
  TrendingUp,
  Sparkles,
  HelpCircle,
  CheckCircle2,
  RefreshCw,
  Send,
} from 'lucide-react';
import { NuancedEmotions, SentimentAnalyzeResult } from '../types';
import { StatusBadge } from '../components/common/StatusBadge';
import { EmotionRadar } from '../components/visualizers/EmotionRadar';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
} from 'recharts';
import { mockTemporalSentiment } from '../services/mockData';
import { api } from '../services/api';

export const SentimentView: React.FC = () => {
  const [inputText, setInputText] = useState(
    'Breakthrough in open-weights reasoning models! The latency reduction and edge quantization performance are astonishing. #AI'
  );
  const [stanceTarget, setStanceTarget] = useState('Artificial Intelligence');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<SentimentAnalyzeResult>({
    text: 'Breakthrough in open-weights reasoning models! The latency reduction and edge quantization performance are astonishing. #AI',
    sentiment: 'positive',
    sentiment_score: 0.82,
    emotions: {
      joy: 0.65,
      optimism: 0.78,
      excitement: 0.82,
      surprise: 0.35,
      anxiety: 0.12,
      sadness: 0.02,
      anger: 0.01,
      disgust: 0.01,
      fear: 0.03,
    },
    primary_emotion: 'excitement',
    irony_score: 0.08,
    is_ironic: false,
    effective_polarity: 'positive',
    irony_inverted: false,
    stance_target: 'Artificial Intelligence',
    stance: 'supportive',
    stance_score: 0.88,
  });

  const handleAnalyze = async () => {
    if (!inputText.trim()) return;
    setIsAnalyzing(true);
    try {
      const data = await api.analyzeSentiment(inputText, stanceTarget);
      setResult(data);
    } catch (e) {
      console.error(e);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const sampleInputs = [
    {
      label: 'Excited Announcement',
      text: 'Thrilled to announce that our team achieved sub-5ms local LLM inference! The community feedback has been extraordinary.',
      target: 'Artificial Intelligence',
    },
    {
      label: 'Sarcastic Critique',
      text: 'Oh absolutely brilliant, another AI benchmark that magically beats human experts by 500% in a cherry-picked footnote. Groundbreaking engineering! 🙄',
      target: 'Artificial Intelligence',
    },
    {
      label: 'Climate Anxiety',
      text: 'Severe summer grid outages and rising ocean temperatures are terrifying. Action is moving far too slowly.',
      target: 'Climate Action',
    },
  ];

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
              Sentiment & Tone
            </span>
            <h2 className="text-xl font-bold text-slate-900">
              Audience Sentiment & Emotion Studio
            </h2>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Analyzes nuanced emotions, detects sarcasm, and evaluates audience stance on key topics.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="px-3 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-800 border border-emerald-200 flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
            AI Emotion Engine Active
          </span>
        </div>
      </div>

      {/* Top Sentiment Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs">
          <div className="text-xs font-medium text-slate-500">Overall Audience Sentiment</div>
          <div className="text-2xl font-bold text-slate-900 mt-2 flex items-center gap-2">
            <span>68% Positive</span>
            <span className="text-xs px-2 py-0.5 bg-emerald-50 text-emerald-700 rounded-full font-semibold">
              Net: +0.48
            </span>
          </div>
          <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden mt-3 flex">
            <div className="bg-emerald-500 h-full" style={{ width: '68%' }} title="68% Positive" />
            <div className="bg-slate-300 h-full" style={{ width: '22%' }} title="22% Neutral" />
            <div className="bg-rose-500 h-full" style={{ width: '10%' }} title="10% Negative" />
          </div>
          <div className="flex justify-between text-[10px] text-slate-400 mt-1.5">
            <span>68% Positive</span>
            <span>22% Neutral</span>
            <span>10% Negative</span>
          </div>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs">
          <div className="text-xs font-medium text-slate-500">Sarcasm & Irony Rate</div>
          <div className="text-2xl font-bold text-slate-900 mt-2 flex items-center gap-2">
            <span>8.4% Detected</span>
            <span className="text-xs px-2 py-0.5 bg-amber-50 text-amber-700 rounded-full font-semibold">
              Calibrated
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-2 leading-relaxed">
            When sarcasm confidence exceeds 85%, apparent positive words are automatically inverted to negative polarity.
          </p>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs">
          <div className="text-xs font-medium text-slate-500">Dominant Nuanced Mood</div>
          <div className="text-2xl font-bold text-slate-900 mt-2 flex items-center gap-2">
            <span>Optimism & Joy</span>
          </div>
          <p className="text-xs text-slate-500 mt-2 leading-relaxed">
            Leading emotional drivers across discussions, followed by surprise and low anxiety.
          </p>
        </div>
      </div>

      {/* 9-Nuanced Emotion Radar */}
      <EmotionRadar emotions={result.emotions} />

      {/* Temporal Sentiment Fluctuation Chart */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-bold text-slate-900">
              Sentiment & Sarcasm Fluctuation Along the Timeline
            </h3>
            <p className="text-xs text-slate-500">Rolling sentiment score and sarcasm rate throughout observation hours</p>
          </div>
          <div className="flex items-center gap-3 text-xs">
            <span className="flex items-center gap-1 text-slate-600 font-medium">
              <span className="w-2.5 h-2.5 rounded-full bg-indigo-600" /> Sentiment Score
            </span>
            <span className="flex items-center gap-1 text-slate-600 font-medium">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-500" /> Sarcasm Rate
            </span>
          </div>
        </div>

        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={mockTemporalSentiment}>
              <XAxis dataKey="bucket" stroke="#94a3b8" fontSize={11} tickLine={false} />
              <YAxis stroke="#94a3b8" fontSize={11} tickLine={false} domain={[-0.2, 1.0]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  borderRadius: '8px',
                  color: '#fff',
                  border: 'none',
                  fontSize: '12px',
                }}
              />
              <Line
                type="monotone"
                dataKey="mean_sentiment_score"
                name="Sentiment Score"
                stroke="#6366f1"
                strokeWidth={2.5}
                dot={{ r: 3 }}
              />
              <Line
                type="monotone"
                dataKey="sarcasm_rate"
                name="Sarcasm Rate"
                stroke="#f59e0b"
                strokeWidth={2}
                strokeDasharray="4 4"
                dot={{ r: 3 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Interactive Live Sentiment Tester (Playground) */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-xs">
        <div className="p-5 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-indigo-600" />
              <h3 className="text-sm font-bold text-slate-900">
                Interactive Live Sentiment & Nuanced Emotion Tester
              </h3>
            </div>
            <p className="text-xs text-slate-500 mt-0.5">
              Paste or type any comment to test real-time emotion classification, sarcasm detection, and topic stance.
            </p>
          </div>
        </div>

        <div className="p-5 space-y-4">
          {/* Quick presets */}
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="text-slate-400 font-medium">Try an example:</span>
            {sampleInputs.map((sample) => (
              <button
                key={sample.label}
                type="button"
                onClick={() => {
                  setInputText(sample.text);
                  setStanceTarget(sample.target);
                }}
                className="px-2.5 py-1 rounded-md bg-slate-100 hover:bg-indigo-50 hover:text-indigo-700 text-slate-700 transition-colors font-medium text-xs"
              >
                {sample.label}
              </button>
            ))}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="md:col-span-2 space-y-2">
              <label className="text-xs font-semibold text-slate-700">Text to Analyze</label>
              <textarea
                rows={3}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Enter social post or user comment..."
                className="w-full p-3 bg-slate-50 border border-slate-200 rounded-lg text-xs text-slate-900 focus:outline-hidden focus:border-indigo-500 focus:bg-white transition-all font-sans"
              />
            </div>

            <div className="space-y-2">
              <label className="text-xs font-semibold text-slate-700">Stance Target (Topic)</label>
              <input
                type="text"
                value={stanceTarget}
                onChange={(e) => setStanceTarget(e.target.value)}
                placeholder="e.g. Artificial Intelligence, Climate..."
                className="w-full p-3 bg-slate-50 border border-slate-200 rounded-lg text-xs text-slate-900 focus:outline-hidden focus:border-indigo-500 focus:bg-white transition-all"
              />

              <button
                type="button"
                onClick={handleAnalyze}
                disabled={isAnalyzing || !inputText.trim()}
                className="w-full mt-2 py-2.5 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white rounded-lg text-xs font-semibold shadow-xs flex items-center justify-center gap-2 transition-colors"
              >
                {isAnalyzing ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    <span>Inferring Emotions...</span>
                  </>
                ) : (
                  <>
                    <Send className="w-3.5 h-3.5" />
                    <span>Run AI Analysis</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Live Inference Output Cards */}
          {result && (
            <div className="mt-4 pt-4 border-t border-slate-100 grid grid-cols-1 sm:grid-cols-3 gap-4">
              {/* Effective Polarity */}
              <div className="p-4 bg-slate-50 rounded-xl border border-slate-200">
                <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
                  Overall Sentiment
                </div>
                <div className="flex items-center gap-2">
                  <StatusBadge type="sentiment" value={result.effective_polarity} />
                  <span className="text-xs font-bold text-slate-700">
                    Score: {result.sentiment_score > 0 ? `+${result.sentiment_score.toFixed(2)}` : result.sentiment_score.toFixed(2)}
                  </span>
                </div>
                {result.irony_inverted && (
                  <p className="mt-2 text-[11px] text-amber-700 bg-amber-50 p-2 rounded border border-amber-200 leading-snug">
                    ⚡ <strong>Sarcasm Inversion:</strong> Surface text appeared positive, but detected high irony ({(result.irony_score * 100).toFixed(0)}%) inverted it to negative.
                  </p>
                )}
              </div>

              {/* Primary Emotion */}
              <div className="p-4 bg-slate-50 rounded-xl border border-slate-200">
                <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
                  Primary Emotion
                </div>
                <div className="text-sm font-bold text-slate-900 capitalize">
                  {result.primary_emotion}
                </div>
                <p className="text-[11px] text-slate-500 mt-1">
                  Confidence: {Math.round((result.emotions[result.primary_emotion] || 0.8) * 100)}%
                </p>
              </div>

              {/* Stance on Target */}
              <div className="p-4 bg-slate-50 rounded-xl border border-slate-200">
                <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
                  Stance on &quot;{result.stance_target || 'Target'}&quot;
                </div>
                <div className="text-sm font-bold text-slate-900 capitalize">
                  {result.stance || 'Supportive'}
                </div>
                <p className="text-[11px] text-slate-500 mt-1">
                  Strength: {result.stance_score ? `${(result.stance_score * 100).toFixed(0)}%` : 'Strong'}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
