import Link from "next/link";
import { PRODUCT_NAME, PRODUCT_NAME_LATIN } from "@/lib/brand";
import styles from "./siteHeader.module.css";

export function SiteHeader({ compact = false }: { compact?: boolean }) {
  return (
    <header className={`${styles.header} ${compact ? styles.headerCompact : ""}`}>
      <Link className={styles.brand} href="/" aria-label={`${PRODUCT_NAME}首页`}>
        <span className={styles.brandSeal} aria-hidden="true">卧</span>
        <span>
          <strong>{PRODUCT_NAME}</strong>
          <small>{PRODUCT_NAME_LATIN}</small>
        </span>
      </Link>
      <nav className={styles.nav} aria-label="主导航">
        <Link href="/#how-it-works">怎么使用</Link>
        <Link href="/#sources">馆藏来源</Link>
        <span className={styles.internalLabel}>内部工具</span>
      </nav>
    </header>
  );
}
