import type { Metadata } from "next";
import {
  COLLECTION_SCOPE,
  CURATOR_TITLE,
  PRODUCT_NAME,
  PRODUCT_NAME_LATIN,
} from "@/lib/brand";
import "./globals.css";

export const metadata: Metadata = {
  title: `${PRODUCT_NAME} ${PRODUCT_NAME_LATIN} · ${CURATOR_TITLE}`,
  description: `和${CURATOR_TITLE}聊几句；它会从${COLLECTION_SCOPE}中寻找线索，为你的问题组织一场有来源可查的 3D 虚拟展览。`,
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
