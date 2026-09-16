import React, { useState } from 'react';
import { HelpCircle } from 'lucide-react';

interface TooltipInfoProps {
  text: string;
  className?: string;
}

export const TooltipInfo: React.FC<TooltipInfoProps> = ({ text, className = '' }) => {
  const [visible, setVisible] = useState(false);

  return (
    <div className={`relative inline-flex items-center ${className}`}>
      <button
        type="button"
        onMouseEnter={() => setVisible(true)}
        onMouseLeave={() => setVisible(false)}
        onClick={() => setVisible(!visible)}
        aria-label="More information"
        className="text-slate-400 hover:text-slate-600 transition-colors focus:outline-hidden"
      >
        <HelpCircle className="w-4 h-4" />
      </button>
      {visible && (
        <div className="absolute z-50 bottom-full mb-1.5 left-1/2 -translate-x-1/2 w-60 p-2.5 bg-slate-900 text-white text-xs rounded-lg shadow-lg pointer-events-none transition-all leading-relaxed">
          {text}
          <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-slate-900" />
        </div>
      )}
    </div>
  );
};
