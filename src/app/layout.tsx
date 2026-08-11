import type { Metadata } from "next";
import {
  COLLECTION_SCOPE,
  CURATOR_TITLE,
  PRODUCT_NAME,
  PRODUCT_NAME_LATIN,
} from "@/lib/brand";
import { DEFAULT_LANGUAGE, LANGUAGE_BOOT_SCRIPT } from "@/lib/i18n";
import "./globals.css";

export const metadata: Metadata = {
  title: `${PRODUCT_NAME} ${PRODUCT_NAME_LATIN} · ${CURATOR_TITLE}`,
  description: `和${CURATOR_TITLE}聊几句；它会从${COLLECTION_SCOPE}中寻找线索，为你的问题组织一场有来源可查的 3D 虚拟展览。`,
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    // The document renders in the default language and the boot script below
    // corrects it during HTML parsing, before first paint. suppressHydration-
    // Warning tells React to accept the DOM the script produced rather than
    // treating the difference as an error.
    <html lang="zh-CN" data-lang={DEFAULT_LANGUAGE} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: LANGUAGE_BOOT_SCRIPT }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
