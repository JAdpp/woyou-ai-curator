import Image from "next/image";
import styles from "./curatorAvatar.module.css";

type CuratorAvatarSize = "xs" | "sm" | "md" | "lg";

export function CuratorAvatar({
  size = "md",
  className = "",
}: {
  size?: CuratorAvatarSize;
  className?: string;
}) {
  return (
    <span
      className={[styles.avatar, styles[size], className].filter(Boolean).join(" ")}
      aria-hidden="true"
    >
      <Image
        src="/brand/yanyuan-avatar-pixel.png"
        alt=""
        width={384}
        height={384}
        sizes="56px"
        unoptimized
        draggable={false}
      />
    </span>
  );
}
