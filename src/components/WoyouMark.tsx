interface WoyouMarkProps {
  className?: string;
}

/**
 * 卧游 as an open handscroll: the central line is both a mountain ridge and
 * the route a visitor follows through an image. The mark deliberately avoids
 * a closed seal outline so it stays light at navigation and favicon sizes.
 */
export function WoyouMark({ className }: WoyouMarkProps) {
  return (
    <svg
      className={className}
      viewBox="0 0 48 48"
      fill="none"
      aria-hidden="true"
      focusable="false"
    >
      <path
        d="M8 14v21M40 14v21"
        stroke="currentColor"
        strokeWidth="2.75"
        strokeLinecap="round"
      />
      <path
        d="M9 31c5 0 6-9 12-9 4 0 5 7 10 7 4 0 6-6 9-11"
        stroke="currentColor"
        strokeWidth="2.75"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M13 35c7-2.2 14 2 23-2"
        stroke="currentColor"
        strokeWidth="2.25"
        strokeLinecap="round"
        opacity="0.56"
      />
      <circle cx="13" cy="31" r="1.8" fill="var(--woyou-mark-accent, #b44532)" />
    </svg>
  );
}
