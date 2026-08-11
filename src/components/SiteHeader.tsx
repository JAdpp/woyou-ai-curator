"use client";

import Link from "next/link";
import { PRODUCT_NAME, PRODUCT_NAME_LATIN } from "@/lib/brand";
import { useLanguage } from "@/lib/useLanguage";
import { LanguageToggle } from "./LanguageToggle";
import { WoyouMark } from "./WoyouMark";
import styles from "./siteHeader.module.css";

/**
 * Site navigation.
 *
 * `tone="onDark"` is for the landing hero, which is a full-height wall of
 * collection images: the bar sits over it rather than above it, so the wall
 * still opens the page at full bleed.
 */
export function SiteHeader({
  compact = false,
  tone = "paper",
  internalLabel = true,
}: {
  compact?: boolean;
  tone?: "paper" | "onDark";
  internalLabel?: boolean;
}) {
  const { language, setLanguage, t } = useLanguage();
  // The wordmark keeps both scripts either way; only which one leads changes,
  // because 卧游 is the product's name, not a string to be translated.
  const lead = language === "en" ? PRODUCT_NAME_LATIN : PRODUCT_NAME;
  const trail = language === "en" ? PRODUCT_NAME : PRODUCT_NAME_LATIN;

  return (
    <header
      className={[
        styles.header,
        compact ? styles.headerCompact : "",
        tone === "onDark" ? styles.headerOnDark : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <Link className={styles.brand} href="/" aria-label={`${lead} ${t.header.home}`}>
        <WoyouMark className={styles.brandMark} />
        <span>
          <strong>{lead}</strong>
          <small>{trail}</small>
        </span>
      </Link>
      <nav className={styles.nav} aria-label={t.header.nav}>
        <Link href="/#how-it-works">{t.header.howItWorks}</Link>
        <Link href="/#sources">{t.header.sources}</Link>
        {internalLabel && (
          <span className={styles.internalLabel}>{t.header.internalTool}</span>
        )}
        <LanguageToggle
          language={language}
          onChange={setLanguage}
          label={t.header.languageGroup}
        />
      </nav>
    </header>
  );
}
