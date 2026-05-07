import { cn } from '@/lib/utils';

export function Card({ children, className, style }: {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <div
      className={cn(className)}
      style={{
        background: 'var(--bg-glass)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius)',
        ...style,
      }}
    >
      {children}
    </div>
  );
}

export function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p style={{
      color: 'var(--text-tertiary)',
      fontSize: 10,
      letterSpacing: '0.10em',
      fontWeight: 600,
      textTransform: 'uppercase',
      marginBottom: 12,
    }}>
      {children}
    </p>
  );
}
