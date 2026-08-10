import { strict as assert } from "node:assert";
import test from "node:test";
import {
  getImageLicenseLabel,
  getLegacyRightsLabel,
  UNSPECIFIED_IMAGE_LICENSE,
} from "./rights";

test("an unknown institution never inherits its legacy rights as an image licence", () => {
  const unknownInstitution = {
    imageLicense: null,
    rights: "CC0 legacy aggregate statement",
  };

  assert.equal(getImageLicenseLabel(unknownInstitution), UNSPECIFIED_IMAGE_LICENSE);
  assert.equal(
    getLegacyRightsLabel(unknownInstitution),
    "CC0 legacy aggregate statement",
  );
});

test("an explicit image licence remains the only image licence display", () => {
  const classifiedObject = { imageLicense: "CC0 1.0", rights: "legacy value" };

  assert.equal(getImageLicenseLabel(classifiedObject), "CC0 1.0");
  assert.equal(getLegacyRightsLabel(classifiedObject), null);
});
