const DETERMINISTIC_FALLBACK_NOTICE =
  "DeepSeek 框架阶段超时，策展文本使用确定性回退；模型字段仅表示配置模型。";

export function generationFallbackNotice(provider?: string): string | null {
  return provider === "deterministic_fallback"
    ? DETERMINISTIC_FALLBACK_NOTICE
    : null;
}

export function generationProviderLabel(provider?: string): string {
  return provider?.trim() || "旧记录未保存";
}
