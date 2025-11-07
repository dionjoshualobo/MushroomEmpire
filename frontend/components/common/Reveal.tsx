"use client";
import { useEffect, useRef, useState } from "react";

interface RevealProps {
  children: React.ReactNode;
  as?: keyof JSX.IntrinsicElements;
  className?: string;
  delayMs?: number;
}

export function Reveal({ children, as = "div", className, delayMs = 0 }: RevealProps) {
  const Cmp: any = as;
  const ref = useRef<HTMLElement | null>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current as Element | null;
    if (!el || typeof IntersectionObserver === "undefined") {
      setVisible(true);
      return;
    }
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          if (delayMs > 0) {
            const t = setTimeout(() => setVisible(true), delayMs);
            return () => clearTimeout(t);
          } else {
            setVisible(true);
          }
        }
      },
      { threshold: 0.12 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [delayMs]);

  return (
    <Cmp ref={ref} className={(visible ? "reveal-visible " : "reveal ") + (className ?? "")}>{children}</Cmp>
  );
}
