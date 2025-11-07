import './globals.css';
import type { ReactNode } from 'react';
import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Nordic Privacy AI',
  description: 'AI-powered GDPR compliance platform for Nordic ecosystems',
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-full bg-gradient-to-br from-brand-50 via-white to-brand-100 text-slate-900 antialiased">
        {children}
      </body>
    </html>
  );
}
