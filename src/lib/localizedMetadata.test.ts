import assert from "node:assert/strict";
import test from "node:test";
import type { ExhibitionItem } from "./types";
import {
  publicInstitutionName,
  publicObjectMetadata,
  publicObjectTitle,
  spokenObjectTitle,
  visitorRoleTag,
} from "./localizedMetadata";

const item = {
  displayTitle: "女子与犬",
  localizedMetadata: {
    creator: "佚名",
    date: "1710年至1720年",
    medium: "象牙水彩",
    culture: "十八世纪意大利",
    institution: "克利夫兰艺术博物馆",
  },
  object: {
    title: "Woman with a Dog",
    titleOriginal: null,
    maker: "Unknown artist",
    date: "1710-1720",
    medium: "watercolor on ivory",
    culture: "Italy",
    institution: "Cleveland Museum of Art",
  },
} as unknown as ExhibitionItem;

test("Chinese public tombstone uses only localized metadata", () => {
  assert.equal(publicObjectTitle(item, "zh-CN"), "女子与犬");
  assert.deepEqual(publicObjectMetadata(item, "zh-CN"), [
    "佚名",
    "1710年至1720年",
    "象牙水彩",
    "十八世纪意大利",
  ]);
  assert.equal(publicInstitutionName(item, "zh-CN"), "克利夫兰艺术博物馆");
});

test("Chinese tombstone keeps English out of metadata but still names the object", () => {
  const legacy = { ...item, displayTitle: "Woman with a Dog", localizedMetadata: undefined };
  assert.equal(publicObjectTitle(legacy, "zh-CN"), "Woman with a Dog");
  assert.equal(spokenObjectTitle(legacy, "zh-CN"), "这件展品");
  assert.deepEqual(publicObjectMetadata(legacy, "zh-CN"), []);
  assert.equal(publicInstitutionName(legacy, "zh-CN"), "");
});

test("role tags use visitor words, never the internal curatorial role", () => {
  const core = { ...item, role: "core_evidence", roleLabel: "核心证据" } as ExhibitionItem;
  const contrast = { ...item, role: "contrast", roleLabel: "对照／其他声音" } as ExhibitionItem;
  assert.equal(visitorRoleTag(core, "zh-CN"), "重点");
  assert.equal(visitorRoleTag(contrast, "zh-CN"), "换个角度");
  assert.equal(visitorRoleTag(contrast, "en"), "Another angle");
});

test("English exhibitions preserve the institution record", () => {
  assert.equal(publicObjectTitle(item, "en"), "女子与犬");
  assert.deepEqual(publicObjectMetadata(item, "en"), [
    "Unknown artist",
    "1710-1720",
    "watercolor on ivory",
    "Italy",
  ]);
  assert.equal(publicInstitutionName(item, "en"), "Cleveland Museum of Art");
});
