import type { MuseumObject } from "./types";

export const UNSPECIFIED_IMAGE_LICENSE = "未逐字段注明（见机构页面）";

type ImageRightsRecord = Pick<MuseumObject, "imageLicense" | "rights">;

/**
 * A legacy object-level rights string is not evidence of an image licence.
 * Keep the two displays separate so future, unknown institutions stay
 * unclassified until their adapter records an explicit field policy.
 */
export function getImageLicenseLabel(
  record: Pick<ImageRightsRecord, "imageLicense">,
): string {
  return record.imageLicense?.trim() || UNSPECIFIED_IMAGE_LICENSE;
}

export function getLegacyRightsLabel(record: ImageRightsRecord): string | null {
  if (record.imageLicense?.trim()) return null;
  return record.rights?.trim() || null;
}
