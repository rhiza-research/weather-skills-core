import json

import pytest
import xarray as xr
from conftest import make_gridded

from weather_skills_core import provenance


def write_store(path, history=None, *, raw=None, fill=1.0):
    """Write a tiny gridded store, optionally stamped with a history chain."""
    ds = make_gridded(fill=fill)
    if raw is not None:
        ds.attrs["weather_skills_history"] = raw
    elif history is not None:
        ds.attrs["weather_skills_history"] = json.dumps(history, sort_keys=True)
    ds.to_zarr(path, mode="w", consolidated=True)
    return path


def entry(**overrides):
    base = {
        "skill": "clip-region",
        "version": "0.1.0",
        "args": {"bbox": "1/2/3/4"},
        "input": {"basename": "in.zarr", "hash": "abc"},
    }
    base.update(overrides)
    return base


class TestHashZarr:
    def test_deterministic(self, tmp_path):
        store = write_store(tmp_path / "a.zarr")
        assert provenance.hash_zarr(store) == provenance.hash_zarr(store)

    def test_content_change_changes_hash(self, tmp_path):
        a = write_store(tmp_path / "a.zarr", fill=1.0)
        before = provenance.hash_zarr(a)
        write_store(tmp_path / "a.zarr", fill=2.0)
        assert provenance.hash_zarr(a) != before

    def test_identical_content_same_hash(self, tmp_path):
        a = write_store(tmp_path / "a.zarr")
        b = write_store(tmp_path / "b.zarr")
        assert provenance.hash_zarr(a) == provenance.hash_zarr(b)


class TestLoadHistory:
    def test_missing_store_is_empty(self, tmp_path):
        assert provenance.load_history(tmp_path / "nope.zarr") == []

    def test_store_without_history_is_empty(self, tmp_path):
        assert provenance.load_history(write_store(tmp_path / "a.zarr")) == []

    def test_valid_history(self, tmp_path):
        chain = [entry()]
        store = write_store(tmp_path / "a.zarr", chain)
        assert provenance.load_history(store) == chain

    def test_rhiza_history_attr_is_not_read(self, tmp_path):
        # A store carrying only a rhiza_history attr has no history.
        ds = make_gridded()
        ds.attrs["rhiza_history"] = json.dumps([entry()], sort_keys=True)
        path = tmp_path / "a.zarr"
        ds.to_zarr(path, mode="w", consolidated=True)
        assert provenance.load_history(path) == []

    def test_permission_error_is_miss(self, tmp_path, monkeypatch):
        import xarray as xr

        store = write_store(tmp_path / "a.zarr", [entry()])

        def denied(*args, **kwargs):
            raise PermissionError("denied")

        monkeypatch.setattr(xr, "open_zarr", denied)
        assert provenance.load_history(store) == []

    def test_os_error_is_miss(self, tmp_path, monkeypatch):
        import xarray as xr

        store = write_store(tmp_path / "a.zarr", [entry()])

        def flaky(*args, **kwargs):
            raise OSError("I/O error")

        monkeypatch.setattr(xr, "open_zarr", flaky)
        assert provenance.load_history(store) == []

    def test_json_object_is_malformed(self, tmp_path, capsys):
        store = write_store(tmp_path / "a.zarr", raw=json.dumps({"skill": "x"}))
        assert provenance.load_history(store) == []
        assert "provenance --check" in capsys.readouterr().err

    def test_non_json_is_malformed(self, tmp_path, capsys):
        store = write_store(tmp_path / "a.zarr", raw="not json at all")
        assert provenance.load_history(store) == []
        assert "provenance --check" in capsys.readouterr().err

    def test_imperfect_entries_pass_through(self, tmp_path):
        # Coercion is array-level only: entries missing keys are not touched.
        chain = [{"unexpected": True}]
        store = write_store(tmp_path / "a.zarr", chain)
        assert provenance.load_history(store) == chain


class TestParseChain:
    def test_valid_array(self):
        chain = [entry()]
        assert provenance.parse_chain(json.dumps(chain)) == chain

    def test_empty_array(self):
        assert provenance.parse_chain("[]") == []

    def test_non_json_raises(self):
        with pytest.raises(ValueError, match="^value is not valid JSON$"):
            provenance.parse_chain("not json at all")

    def test_none_raises_not_valid_json(self):
        with pytest.raises(ValueError, match="^value is not valid JSON$"):
            provenance.parse_chain(None)

    def test_json_object_raises_not_array(self):
        with pytest.raises(ValueError, match="^value is not a JSON array$"):
            provenance.parse_chain(json.dumps({"skill": "x"}))

    def test_json_scalar_raises_not_array(self):
        with pytest.raises(ValueError, match="^value is not a JSON array$"):
            provenance.parse_chain("42")


class TestCoerceChain:
    def test_valid_array_passes_through(self, capsys):
        chain = [entry()]
        assert provenance.coerce_chain(json.dumps(chain), "a.zarr") == chain
        assert capsys.readouterr().err == ""

    def test_empty_array_passes_through(self, capsys):
        assert provenance.coerce_chain("[]", "a.zarr") == []
        assert capsys.readouterr().err == ""

    def test_imperfect_entries_pass_through(self):
        # Coercion is array-level only: entries missing keys are not touched.
        chain = [{"unexpected": True}]
        assert provenance.coerce_chain(json.dumps(chain), "a.zarr") == chain

    def test_non_json_warns_and_returns_none(self, capsys):
        label = "plot.png (weather_skills_history_a)"
        assert provenance.coerce_chain("not json at all", label) is None
        assert capsys.readouterr().err == (
            "ignoring malformed weather_skills_history on plot.png "
            "(weather_skills_history_a); run `provenance --check` for details\n"
        )

    def test_json_object_warns_and_returns_none(self, capsys):
        assert provenance.coerce_chain(json.dumps({"skill": "x"}), "a.zarr") is None
        err = capsys.readouterr().err
        assert err == (
            "ignoring malformed weather_skills_history on a.zarr; "
            "run `provenance --check` for details\n"
        )

    def test_json_scalar_warns_and_returns_none(self, capsys):
        assert provenance.coerce_chain("42", "a.zarr") is None
        assert "provenance --check" in capsys.readouterr().err


class TestValidateChain:
    def test_valid_chain(self):
        violations, notes = provenance.validate_chain([entry(input=None), entry()], "h")
        assert violations == []
        assert notes == []

    def test_non_list_chain(self):
        violations, notes = provenance.validate_chain({"skill": "x"}, "h")
        assert violations == ["h: value is not a JSON array"]
        assert notes == []

    def test_non_dict_entry(self):
        violations, _ = provenance.validate_chain(["nope"], "h")
        assert violations == ["h[0]: entry is not an object"]

    def test_missing_required_keys(self):
        violations, _ = provenance.validate_chain([{}], "h")
        assert violations == [
            "h[0]: missing required key 'skill'",
            "h[0]: missing required key 'version'",
            "h[0]: missing required key 'args'",
            "h[0]: missing required key 'input'",
        ]

    def test_mistyped_fields(self):
        bad = {"skill": 1, "version": 2, "args": [], "input": 3}
        violations, _ = provenance.validate_chain([bad], "h")
        assert violations == [
            "h[0].skill: must be a string",
            "h[0].version: must be a string",
            "h[0].args: must be an object",
            "h[0].input: must be null, an object, or an array of objects",
        ]

    def test_empty_skill_string(self):
        violations, _ = provenance.validate_chain([entry(skill="")], "h")
        assert violations == ["h[0].skill: must be a non-empty string"]

    def test_unknown_entry_key_is_note(self):
        violations, notes = provenance.validate_chain([dict(entry(), extra=1)], "h")
        assert violations == []
        assert notes == ["h[0]: unknown key 'extra'"]

    def test_violation_location_uses_entry_index(self):
        violations, _ = provenance.validate_chain([entry(), entry(version=1)], "h")
        assert violations == ["h[1].version: must be a string"]

    def test_input_dict_missing_keys(self):
        violations, _ = provenance.validate_chain([entry(input={})], "h")
        assert violations == [
            "h[0].input: missing required key 'basename'",
            "h[0].input: missing required key 'hash'",
        ]

    def test_input_dict_mistyped_values(self):
        violations, _ = provenance.validate_chain([entry(input={"basename": 1, "hash": 2})], "h")
        assert violations == [
            "h[0].input.basename: must be a string",
            "h[0].input.hash: must be a string",
        ]

    def test_input_list_items_located(self):
        e = entry(input=[{"basename": "a.zarr", "hash": "x"}, "bad"])
        violations, _ = provenance.validate_chain([e], "h")
        assert violations == ["h[0].input[1]: input entry is not an object"]

    def test_input_item_unknown_key_is_note(self):
        e = entry(input=[{"basename": "a.zarr", "hash": "x", "note": 1}])
        violations, notes = provenance.validate_chain([e], "h")
        assert violations == []
        assert notes == ["h[0].input[0]: unknown key 'note'"]

    def test_multi_input_with_histories_is_valid(self):
        e = entry(
            input=[
                {"basename": "a.zarr", "hash": "ha", "history": []},
                {"basename": "b.zarr", "hash": "hb", "history": [entry(input=None)]},
            ]
        )
        violations, notes = provenance.validate_chain([e], "h")
        assert violations == []
        assert notes == []

    def test_nested_history_recursion(self):
        nested = [{"skill": "", "version": "0.1.0", "args": {}, "input": None}]
        e = entry(input=[{"basename": "a.zarr", "hash": "x", "history": nested}])
        violations, _ = provenance.validate_chain([e], "h")
        assert violations == ["h[0].input[0].history[0].skill: must be a non-empty string"]

    def test_nested_history_non_array(self):
        e = entry(input=[{"basename": "a.zarr", "hash": "x", "history": "bad"}])
        violations, _ = provenance.validate_chain([e], "h")
        assert violations == ["h[0].input[0].history: value is not a JSON array"]


class TestEntryConstruction:
    def test_input_ref_with_hash(self, tmp_path):
        store = write_store(tmp_path / "a.zarr")
        ref = provenance.input_ref(store)
        assert ref["basename"] == "a.zarr"
        assert ref["hash"] == provenance.hash_zarr(store)

    def test_input_ref_without_hash(self, tmp_path):
        ref = provenance.input_ref(write_store(tmp_path / "a.zarr"), include_hash=False)
        assert ref == {"basename": "a.zarr"}

    def test_multi_input_ref(self, tmp_path):
        a = write_store(tmp_path / "a.zarr")
        b = write_store(tmp_path / "b.zarr", [entry()])
        refs = provenance.multi_input_ref([a, b], [[], [entry()]])
        assert [r["basename"] for r in refs] == ["a.zarr", "b.zarr"]
        assert refs[0]["history"] == []
        assert refs[1]["history"] == [entry()]
        assert all("hash" in r for r in refs)

    def test_build_entry_fetcher(self):
        e = provenance.build_entry("chirps-fetch", "0.1.0", {"start": "2026-01-01"}, None)
        assert e["input"] is None
        assert "reference_inputs" not in e

    def test_build_entry_reference_inputs_sibling(self, tmp_path):
        refs = provenance.reference_ref([write_store(tmp_path / "grid.zarr")])
        e = provenance.build_entry("downscale", "0.1.0", {}, {"basename": "a.zarr"}, refs)
        assert e["reference_inputs"][0]["basename"] == "grid.zarr"
        assert "hash" in e["reference_inputs"][0]


class TestStampZarr:
    def test_stamp_sets_sorted_json_and_clears_encoding(self, tmp_path):
        store = write_store(tmp_path / "a.zarr")
        ds = xr.open_zarr(store, consolidated=False)
        ds["precip"].encoding["chunks"] = (1, 1, 1)
        chain = [entry()]
        provenance.stamp_zarr(ds, chain)
        assert ds.attrs["weather_skills_history"] == json.dumps(chain, sort_keys=True)
        assert all(ds[v].encoding == {} for v in ds.variables)

    def test_set_source_names_the_data_product(self):
        ds = make_gridded()
        assert provenance.set_source(ds, "oisst") is ds
        assert ds.attrs["weather_skills_source"] == "oisst"

    def test_stamp_leaves_a_body_set_source_alone(self):
        # The fetcher sets it; the stamp must not clear or overwrite it, or it
        # would never reach the written store.
        ds = provenance.set_source(make_gridded(), "oisst")
        provenance.stamp_zarr(ds, [])
        assert ds.attrs["weather_skills_source"] == "oisst"

    def test_stamp_leaves_unrelated_attrs_untouched(self):
        # rhiza_* attrs are ordinary opaque attrs: no migration, no removal.
        ds = make_gridded()
        ds.attrs["rhiza_source"] = "chirps"
        provenance.stamp_zarr(ds, [])
        assert ds.attrs["rhiza_source"] == "chirps"
        assert "weather_skills_source" not in ds.attrs


class TestRestampZarr:
    def test_history_rewritten_for_both_readers(self, tmp_path):
        store = write_store(tmp_path / "a.zarr", [entry(args={"end": "2026-01-31"})])
        new_chain = [entry(args={"end": "2026-01-02"})]
        provenance.restamp_zarr(store, new_chain)
        assert provenance.load_history(store) == new_chain
        consolidated = xr.open_zarr(store, consolidated=True)
        assert json.loads(consolidated.attrs["weather_skills_history"]) == new_chain

    def test_data_and_other_attrs_untouched(self, tmp_path):
        ds = make_gridded(fill=3.0)
        ds.attrs["weather_skills_source"] = "toy"
        path = tmp_path / "a.zarr"
        ds.to_zarr(path, mode="w", consolidated=True)
        provenance.restamp_zarr(path, [entry()])
        after = xr.open_zarr(path, consolidated=True)
        assert after.attrs["weather_skills_source"] == "toy"
        assert float(after["precip"].values.max()) == 3.0


class TestPngMetadata:
    def test_single_unlabeled(self):
        chain = [entry()]
        md = provenance.png_metadata([(None, chain)])
        assert md["weather_skills_history"] == json.dumps(chain, sort_keys=True)
        assert md["Software"] == "forecasting-skills"

    def test_suffixed_labels(self):
        md = provenance.png_metadata([("a", []), ("b", [entry()])])
        assert set(md) == {"weather_skills_history_a", "weather_skills_history_b", "Software"}

    def test_semantic_labels(self):
        md = provenance.png_metadata([("forecast", []), ("mclimate", [])])
        assert "weather_skills_history_forecast" in md
        assert "weather_skills_history_mclimate" in md

    def test_custom_software(self):
        assert provenance.png_metadata([(None, [])], software="acme")["Software"] == "acme"


class TestCacheHitFetcher:
    def test_hit_on_matching_first_entry(self, tmp_path):
        e = entry(input=None)
        out = write_store(tmp_path / "out.zarr", [e])
        assert provenance.cache_hit(out, e, fetcher=True)

    def test_missing_store_is_miss(self, tmp_path):
        assert not provenance.cache_hit(tmp_path / "out.zarr", entry(input=None), fetcher=True)

    def test_no_history_is_miss(self, tmp_path):
        out = write_store(tmp_path / "out.zarr")
        assert not provenance.cache_hit(out, entry(input=None), fetcher=True)

    @pytest.mark.parametrize(
        "change",
        [{"version": "0.2.0"}, {"args": {"bbox": "9/9/9/9"}}, {"skill": "other"}],
    )
    def test_changed_field_is_miss(self, tmp_path, change):
        out = write_store(tmp_path / "out.zarr", [entry(input=None)])
        assert not provenance.cache_hit(out, entry(input=None, **change), fetcher=True)

    def test_first_entry_position(self, tmp_path):
        # A fetcher hit keys on history[0] even when later entries exist.
        e = entry(input=None)
        out = write_store(tmp_path / "out.zarr", [e, entry(skill="clip-region")])
        assert provenance.cache_hit(out, e, fetcher=True)

    def test_non_dict_first_entry_is_miss(self, tmp_path):
        out = write_store(tmp_path / "out.zarr", ["junk"])
        assert not provenance.cache_hit(out, entry(input=None), fetcher=True)

    def test_completeness_probe_rejects_hit(self, tmp_path, capsys):
        e = entry(input=None)
        out = write_store(tmp_path / "out.zarr", [e])
        assert not provenance.cache_hit(out, e, fetcher=True, completeness_probe=lambda p: False)
        assert "incomplete" in capsys.readouterr().err

    def test_completeness_probe_accepts_hit(self, tmp_path):
        e = entry(input=None)
        out = write_store(tmp_path / "out.zarr", [e])
        probed = []
        assert provenance.cache_hit(
            out, e, fetcher=True, completeness_probe=lambda p: probed.append(p) or True
        )
        assert probed == [out]

    def test_probe_not_called_when_entry_mismatches(self, tmp_path):
        out = write_store(tmp_path / "out.zarr", [entry(input=None)])
        probed = []
        provenance.cache_hit(
            out,
            entry(input=None, version="9.9.9"),
            fetcher=True,
            completeness_probe=lambda p: probed.append(p) or True,
        )
        assert probed == []


class TestCacheHitChained:
    def upstream(self):
        return [entry(skill="chirps-fetch", input=None)]

    def test_hit(self, tmp_path):
        e = entry()
        out = write_store(tmp_path / "out.zarr", self.upstream() + [e])
        assert provenance.cache_hit(out, e, self.upstream())

    def test_upstream_mismatch_is_miss(self, tmp_path):
        e = entry()
        out = write_store(tmp_path / "out.zarr", self.upstream() + [e])
        other = [entry(skill="imerg-fetch", input=None)]
        assert not provenance.cache_hit(out, e, other)

    def test_chain_length_mismatch_is_miss(self, tmp_path):
        e = entry()
        out = write_store(tmp_path / "out.zarr", [e])
        assert not provenance.cache_hit(out, e, self.upstream())

    def test_hash_change_is_miss(self, tmp_path):
        e = entry()
        out = write_store(tmp_path / "out.zarr", self.upstream() + [e])
        changed = entry(input={"basename": "in.zarr", "hash": "different"})
        assert not provenance.cache_hit(out, changed, self.upstream())

    def test_hash_ignored_when_compare_disabled(self, tmp_path):
        e = entry()
        out = write_store(tmp_path / "out.zarr", self.upstream() + [e])
        changed = entry(input={"basename": "in.zarr"})
        assert provenance.cache_hit(out, changed, self.upstream(), compare_hash=False)

    def test_basename_change_is_miss_even_without_hash(self, tmp_path):
        e = entry()
        out = write_store(tmp_path / "out.zarr", self.upstream() + [e])
        changed = entry(input={"basename": "renamed.zarr"})
        assert not provenance.cache_hit(out, changed, self.upstream(), compare_hash=False)

    def test_non_dict_tail_entry_is_miss(self, tmp_path):
        out = write_store(tmp_path / "out.zarr", self.upstream() + ["junk"])
        assert not provenance.cache_hit(out, entry(), self.upstream())

    def test_non_dict_recorded_input_is_miss(self, tmp_path):
        out = write_store(tmp_path / "out.zarr", self.upstream() + [entry(input="junk")])
        assert not provenance.cache_hit(out, entry(), self.upstream())

    def test_non_list_recorded_input_against_multi_entry_is_miss(self, tmp_path):
        out = write_store(tmp_path / "out.zarr", [entry(input="junk")])
        changed = entry(input=[{"basename": "a.zarr", "hash": "ha", "history": []}])
        assert not provenance.cache_hit(out, changed, [])

    def test_non_dict_item_in_recorded_input_list_is_miss(self, tmp_path):
        out = write_store(tmp_path / "out.zarr", [entry(input=["junk"])])
        changed = entry(input=[{"basename": "a.zarr", "hash": "ha", "history": []}])
        assert not provenance.cache_hit(out, changed, [])

    def test_wrong_shaped_basename_is_miss(self, tmp_path):
        recorded = entry(input={"basename": ["in.zarr"], "hash": "abc"})
        out = write_store(tmp_path / "out.zarr", self.upstream() + [recorded])
        assert not provenance.cache_hit(out, entry(), self.upstream())

    def test_multi_input_hit(self, tmp_path):
        inputs = [
            {"basename": "a.zarr", "hash": "ha", "history": []},
            {"basename": "b.zarr", "hash": "hb", "history": self.upstream()},
        ]
        e = entry(input=inputs)
        out = write_store(tmp_path / "out.zarr", [e])
        assert provenance.cache_hit(out, e, [])

    def test_multi_input_hash_change_is_miss(self, tmp_path):
        inputs = [{"basename": "a.zarr", "hash": "ha", "history": []}]
        out = write_store(tmp_path / "out.zarr", [entry(input=inputs)])
        changed = entry(input=[{"basename": "a.zarr", "hash": "other", "history": []}])
        assert not provenance.cache_hit(out, changed, [])

    def test_multi_input_branch_history_change_is_miss(self, tmp_path):
        inputs = [{"basename": "a.zarr", "hash": "ha", "history": []}]
        out = write_store(tmp_path / "out.zarr", [entry(input=inputs)])
        changed = entry(input=[{"basename": "a.zarr", "hash": "ha", "history": self.upstream()}])
        assert not provenance.cache_hit(out, changed, [])

    def test_multi_input_count_change_is_miss(self, tmp_path):
        inputs = [{"basename": "a.zarr", "hash": "ha", "history": []}]
        out = write_store(tmp_path / "out.zarr", [entry(input=inputs)])
        changed = entry(input=inputs + [{"basename": "b.zarr", "hash": "hb", "history": []}])
        assert not provenance.cache_hit(out, changed, [])

    def test_reference_inputs_change_is_miss(self, tmp_path):
        refs = [{"basename": "grid.zarr", "hash": "g1"}]
        e = dict(entry(), reference_inputs=refs)
        out = write_store(tmp_path / "out.zarr", [e])
        changed = dict(entry(), reference_inputs=[{"basename": "grid.zarr", "hash": "g2"}])
        assert provenance.cache_hit(out, e, [])
        assert not provenance.cache_hit(out, changed, [])

    def test_reference_inputs_absent_on_both_is_hit(self, tmp_path):
        e = entry()
        out = write_store(tmp_path / "out.zarr", [e])
        assert provenance.cache_hit(out, e, [])

    def test_completeness_probe_rejects_hit(self, tmp_path, capsys):
        e = entry()
        out = write_store(tmp_path / "out.zarr", self.upstream() + [e])
        assert not provenance.cache_hit(out, e, self.upstream(), completeness_probe=lambda p: False)
        err = capsys.readouterr().err
        assert "incomplete" in err
        assert "recomputing" in err

    def test_completeness_probe_accepts_hit(self, tmp_path):
        e = entry()
        out = write_store(tmp_path / "out.zarr", self.upstream() + [e])
        probed = []
        assert provenance.cache_hit(
            out, e, self.upstream(), completeness_probe=lambda p: probed.append(p) or True
        )
        assert probed == [out]

    def test_probe_not_called_when_entry_mismatches(self, tmp_path):
        out = write_store(tmp_path / "out.zarr", self.upstream() + [entry()])
        probed = []
        provenance.cache_hit(
            out,
            entry(version="9.9.9"),
            self.upstream(),
            completeness_probe=lambda p: probed.append(p) or True,
        )
        assert probed == []


class TestMakeCompletenessProbe:
    def store(self, tmp_path, ds=None, *, time_chunks=None):
        ds = ds if ds is not None else make_gridded()
        encoding = None
        if time_chunks is not None:
            sizes = ds["precip"].shape
            encoding = {"precip": {"chunks": (time_chunks, *sizes[1:])}}
        path = tmp_path / "out.zarr"
        ds.to_zarr(path, mode="w", consolidated=True, encoding=encoding)
        return path

    def corrupt_chunk(self, store, name):
        """Overwrite one named chunk file with undecodable bytes."""
        chunk = store / "precip" / "c" / name
        assert chunk.is_file()
        chunk.write_bytes(b"not a chunk")

    def test_complete_store_with_named_variable(self, tmp_path):
        probe = provenance.make_completeness_probe("precip")
        assert probe(self.store(tmp_path)) is True

    def test_unknown_variable_is_a_miss(self, tmp_path):
        probe = provenance.make_completeness_probe("sst")
        assert probe(self.store(tmp_path)) is False

    def test_missing_store_is_a_miss(self, tmp_path):
        probe = provenance.make_completeness_probe("precip")
        assert probe(tmp_path / "nope.zarr") is False

    def test_unreadable_store_is_a_miss(self, tmp_path):
        path = tmp_path / "not-a-zarr"
        path.mkdir()
        (path / "zarr.json").write_text("garbage")
        probe = provenance.make_completeness_probe("precip")
        assert probe(path) is False

    def test_corrupt_chunk_is_a_miss(self, tmp_path):
        store = self.store(tmp_path)
        self.corrupt_chunk(store, "0/0/0")
        probe = provenance.make_completeness_probe("precip")
        assert probe(store) is False

    def test_probe_all_variables_by_default(self, tmp_path):
        probe = provenance.make_completeness_probe()
        assert probe(self.store(tmp_path)) is True

    def test_store_with_no_data_variables_is_a_miss(self, tmp_path):
        import numpy as np

        ds = xr.Dataset(coords={"time": np.array(["2026-01-01"], dtype="datetime64[ns]")})
        probe = provenance.make_completeness_probe()
        assert probe(self.store(tmp_path, ds)) is False

    def test_variable_list_subset_and_superset(self, tmp_path):
        store = self.store(tmp_path)
        assert provenance.make_completeness_probe(["precip"])(store) is True
        assert provenance.make_completeness_probe(["precip", "sst"])(store) is False

    def test_empty_dimension_is_a_miss(self, tmp_path):
        ds = make_gridded(lats=())
        probe = provenance.make_completeness_probe("precip")
        assert probe(self.store(tmp_path, ds)) is False

    def test_scalar_variable_reads_back(self, tmp_path):
        ds = make_gridded()
        ds["count"] = xr.DataArray(3)
        probe = provenance.make_completeness_probe(["precip", "count"])
        assert probe(self.store(tmp_path, ds)) is True

    def test_callable_spec_reads_the_run_context(self, tmp_path):
        from types import SimpleNamespace

        store = self.store(tmp_path)
        probe = provenance.make_completeness_probe(lambda context: context.args.variable)
        ok = SimpleNamespace(args=SimpleNamespace(variable=["precip"]))
        missing = SimpleNamespace(args=SimpleNamespace(variable=["sst"]))
        assert probe(store, context=ok) is True
        assert probe(store, context=missing) is False

    def test_callable_spec_none_probes_everything(self, tmp_path):
        from types import SimpleNamespace

        probe = provenance.make_completeness_probe(lambda context: context.args.variable)
        context = SimpleNamespace(args=SimpleNamespace(variable=None))
        assert probe(self.store(tmp_path), context=context) is True

    def test_raising_callable_spec_propagates(self, tmp_path):
        # A raising spec is a skill bug: it must crash loudly, never read as
        # a cache miss that silently recomputes on every run.
        probe = provenance.make_completeness_probe(lambda context: context.args.variable)
        with pytest.raises(AttributeError):
            probe(self.store(tmp_path), context=None)

    def test_store_read_failure_with_callable_spec_is_a_miss(self, tmp_path):
        # The spec resolves fine; the unreadable store stays a plain miss.
        from types import SimpleNamespace

        path = tmp_path / "not-a-zarr"
        path.mkdir()
        (path / "zarr.json").write_text("garbage")
        probe = provenance.make_completeness_probe(lambda context: context.args.variable)
        context = SimpleNamespace(args=SimpleNamespace(variable=["precip"]))
        assert probe(path, context=context) is False

    def test_check_time_reads_the_last_slice(self, tmp_path):
        # Two single-step time chunks; corrupting the LAST one is caught only
        # by the check_time corner (index -1 along time), not the default
        # first-corner read.
        store = self.store(tmp_path, time_chunks=1)
        self.corrupt_chunk(store, "1/0/0")
        assert provenance.make_completeness_probe("precip")(store) is True
        probe = provenance.make_completeness_probe("precip", check_time="time")
        assert probe(store) is False

    def test_check_time_accepts_a_complete_store(self, tmp_path):
        probe = provenance.make_completeness_probe("precip", check_time="time")
        assert probe(self.store(tmp_path)) is True

    def test_check_time_missing_coord_raises(self, tmp_path):
        # A check_time coord absent from the store is a misconfiguration, loud
        # like the other unsupported-time cases, not a silent incompleteness
        # miss.
        ds = make_gridded().rename({"time": "t"})
        probe = provenance.make_completeness_probe("precip", check_time="time")
        with pytest.raises(ValueError, match="absent from the store"):
            probe(self.store(tmp_path, ds))

    def test_check_time_nat_is_a_miss(self, tmp_path):
        import numpy as np

        ds = make_gridded()
        times = ds["time"].values.copy()
        times[-1] = np.datetime64("NaT", "ns")
        ds = ds.assign_coords(time=times)
        probe = provenance.make_completeness_probe("precip", check_time="time")
        assert probe(self.store(tmp_path, ds)) is False

    def test_check_time_non_increasing_is_a_miss(self, tmp_path):
        ds = make_gridded()
        ds = ds.assign_coords(time=ds["time"].values[::-1])
        probe = provenance.make_completeness_probe("precip", check_time="time")
        assert probe(self.store(tmp_path, ds)) is False

    def test_check_time_cftime_coordinate_raises(self, tmp_path):
        # A non-standard calendar decodes to cftime objects, which check_time
        # cannot verify. A complete store must never read as a permanent
        # miss, so the unsupported representation raises instead.
        times = xr.date_range("2026-01-01", periods=2, freq="D", calendar="noleap")
        ds = make_gridded().assign_coords(time=times)
        store = self.store(tmp_path, ds)
        probe = provenance.make_completeness_probe("precip", check_time="time")
        with pytest.raises(ValueError, match="datetime64"):
            probe(store)

    def test_check_time_numeric_coordinate_raises(self, tmp_path):
        import numpy as np

        ds = make_gridded()
        ds = ds.assign_coords(time=np.arange(ds.sizes["time"]))
        store = self.store(tmp_path, ds)
        probe = provenance.make_completeness_probe("precip", check_time="time")
        with pytest.raises(ValueError, match="datetime64"):
            probe(store)

    def test_check_time_scalar_coordinate_raises(self, tmp_path):
        # A forecast-style scalar init time is not a dimension coordinate;
        # check_time cannot verify it, and a complete store must never read
        # as a permanent miss, so the unsupported shape raises instead.
        ds = make_gridded().isel(time=0)
        store = self.store(tmp_path, ds)
        probe = provenance.make_completeness_probe("precip", check_time="time")
        with pytest.raises(ValueError, match="dimension coordinate"):
            probe(store)

    def test_unconsolidated_store_is_not_a_permanent_miss(self, tmp_path):
        # open_zarr(consolidated=True) raises when a valid store carries no
        # consolidated metadata; the probe falls back to consolidated=False
        # rather than concluding a miss.
        path = tmp_path / "out.zarr"
        make_gridded().to_zarr(path, mode="w", consolidated=False)
        probe = provenance.make_completeness_probe("precip")
        assert probe(path) is True

    def test_unconsolidated_store_with_check_time(self, tmp_path):
        path = tmp_path / "out.zarr"
        make_gridded().to_zarr(path, mode="w", consolidated=False)
        probe = provenance.make_completeness_probe("precip", check_time="time")
        assert probe(path) is True

    def test_probe_signature_opts_into_the_run_context(self):
        import inspect

        probe = provenance.make_completeness_probe("precip")
        assert "context" in inspect.signature(probe).parameters
