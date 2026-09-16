import React from 'react';

export const LoadingSkeleton: React.FC<{ rows?: number; className?: string }> = ({
  rows = 3,
  className = '',
}) => {
  return (
    <div className={`space-y-3 animate-pulse ${className}`}>
      <div className="h-4 bg-slate-200 rounded w-3/4" />
      {Array.from({ length: rows - 1 }).map((_, i) => (
        <div key={i} className="h-4 bg-slate-100 rounded w-full" />
      ))}
    </div>
  );
};
