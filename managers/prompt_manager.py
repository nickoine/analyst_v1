# Built-in
from __future__ import annotations
from typing import Any
from pathlib import Path

# External
import importlib, json, yaml

# Internal
from managers.dto import AgentProfile


class PromptManager:
    """Loads agent profile, instructions, assistant and exposes them by agent id."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._profiles: dict[str, AgentProfile] = {}

    def load_profiles(self) -> None:
        """
        Scan the directory and load all .yaml / .yml files.
        """
        self._profiles.clear()

        if not self.directory.exists():
            raise FileNotFoundError(f"Profiles directory not found: {self.directory}")

        if not self.directory.is_dir():
            raise NotADirectoryError(f"Expected directory, got: {self.directory}")

        loaded_stems: set[str] = set()

        for path in sorted(self.directory.glob("*.yaml")):
            profile = self._load_profile(path)
            self._profiles[profile.agent_id] = profile
            loaded_stems.add(path.stem)

        for path in sorted(self.directory.glob("*.yml")):
            if path.stem in loaded_stems:
                continue
            profile = self._load_profile(path)
            self._profiles[profile.agent_id] = profile

    def get(self, agent_id: str) -> AgentProfile | None:
        """
        Return one profile by agent id.
        """
        return self._profiles.get(agent_id)

    def all(self) -> list[AgentProfile]:
        """
        Return all loaded profiles.
        """
        return list(self._profiles.values())

    def require(self, agent_id: str) -> AgentProfile:
        """Return one loaded profile or raise if it does not exist."""
        profile = self.get(agent_id)
        if profile is None:
            raise ValueError(f"Agent profile '{agent_id}' not found")
        return profile

    def _load_profile(self, path: Path) -> AgentProfile:
        """
        Load one YAML file into AgentProfile.
        """
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        if not isinstance(raw, dict):
            raise ValueError(f"{path}: YAML root must be a dict")

        profile_raw = raw.get("agentProfile")
        if not isinstance(profile_raw, dict):
            raise ValueError(f"{path}: missing or invalid 'agentProfile'")

        output_contract_raw = profile_raw.get("output_contract")
        agent_id = path.stem

        return AgentProfile(
            agent_id=agent_id,
            system_content=self._render_text(profile_raw.get("system_content")),
            user_content=self._render_text(profile_raw.get("user_content")),
            actions=self._normalize_actions(profile_raw.get("actions")),
            instructions=self._load_xml(path, agent_id, "instructions"),
            assistant=self._load_xml(path, agent_id, "assistant"),
            output_contract=self._resolve_output_contract(output_contract_raw, path),
            output_reference=self._extract_output_reference(output_contract_raw, path),
            knowledge=self._render_text(profile_raw.get("knowledge")),
            memory=self._render_text(profile_raw.get("memory")),
            evaluators=self._render_text(profile_raw.get("evaluators")),
            reasoners=self._render_text(profile_raw.get("reasoners")),
            planners=self._render_text(profile_raw.get("planners")),
            feedback=self._render_text(profile_raw.get("feedback")),

            reasoner_effort=self._render_text(profile_raw.get("reasoner_effort")),
            reasoner_summary=self._render_text(profile_raw.get("reasoner_summary")),
            prompt_cache=self._render_text(profile_raw.get("prompt_cache")),

            source_path=path,
        )

    @staticmethod
    def _load_xml(profile_path: Path, agent_id: str, kind: str) -> str:
        """
        Generic loader for agent XML artifacts.

        Args:
            profile_path: Path to the YAML profile.
            agent_id: Agent identifier (e.g. "analyzer").
            kind: XML type ("instructions", "assistant", etc.)

        Returns:
            XML content as raw string.
        """
        xml_path = profile_path.parent / f"{agent_id}_{kind}.xml"

        if not xml_path.exists():
            raise FileNotFoundError(
                f"{profile_path}: expected sibling {kind} file "
                f"'{agent_id}_{kind}.xml', not found at {xml_path}"
            )

        if not xml_path.is_file():
            raise ValueError(
                f"{profile_path}: {kind} path is not a file: {xml_path}"
            )

        return xml_path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _render_text(value: Any) -> str:
        """
        Render nested YAML values into text and wrap the result in fenced delimiters.

        Rules:
        - None -> ""
        - str -> stripped string
        - dict/list -> YAML-formatted text
        - other -> str(value)

        Non-empty results are returned as:

        ```
        value
        ```
        """
        if value is None:
            return ""

        if isinstance(value, str):
            rendered = value.strip()
        elif isinstance(value, (dict, list)):
            rendered = yaml.safe_dump(
                value,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            ).strip()
        else:
            rendered = str(value).strip()

        if not rendered:
            return ""

        return f"{rendered}"

    def _extract_output_reference(
            self, value: Any, profile_path: Path
    ) -> str:
        """
        Load the semantic reference example from output_contract.examples and
        return it as toon-encoded fenced text.

        Expected YAML shape:
            output_contract:
              examples:
                - format: JSON
                  path: agent/analyzer_output.json
                  request_binding: instructions
                  purpose: semantic_reference
        """
        if value is None:
            return ""

        if not isinstance(value, dict):
            raise ValueError(f"{profile_path}: 'output_contract' must be a dict")

        examples = value.get("examples")
        if examples is None:
            return ""

        if not isinstance(examples, list):
            raise ValueError(f"{profile_path}: 'output_contract.examples' must be a list")

        example = self._find_semantic_reference_example(examples, profile_path)
        if example is None:
            return ""

        example_format = example.get("format")
        example_path = example.get("path")

        if example_format != "json":
            raise ValueError(
                f"{profile_path}: semantic reference example format must be 'json'"
            )

        if not isinstance(example_path, str) or not example_path.strip():
            raise ValueError(
                f"{profile_path}: semantic reference example must have a non-empty 'path'"
            )

        data = self._load_example_file(example_path.strip(), profile_path)
        return data

    def _load_example_file(
            self, relative_path: str, profile_path: Path
    ) -> Any:
        """
        Load an example output reference file relative to the profile directory.

        Current supports:
        - .json
        - .yaml / .yml
        """
        file_path = self.directory / relative_path

        if not file_path.exists():
            raise FileNotFoundError(
                f"{profile_path}: output reference file not found: {file_path}"
            )

        suffix = file_path.suffix.lower()
        content = file_path.read_text(encoding="utf-8")

        if suffix == ".json":
            return json.loads(content)

        if suffix in {".yaml", ".yml"}:
            return yaml.safe_load(content)

        raise ValueError(
            f"{profile_path}: unsupported output reference format '{suffix}' for file {file_path}"
        )

    @staticmethod
    def _resolve_output_contract(
            value: Any, profile_path: Path
    ) -> Any:
        """
        Resolve the authoritative schema object from:

        output_contract.schema.module
        output_contract.schema.name

        Example:
            module: output_contract.analyzer
            name: AnalyzerHandoff

        Returns:
            The actual Python object, e.g. output_contract.analyzer.AnalyzerHandoff
        """
        if value is None:
            return None

        if not isinstance(value, dict):
            raise ValueError(f"{profile_path}: 'output_contract' must be a dict")

        schema = value.get("schema")
        if not isinstance(schema, dict):
            raise ValueError(f"{profile_path}: 'output_contract.schema' must be a dict")

        request_binding = schema.get("request_binding")
        module_name = schema.get("module")
        object_name = schema.get("name")


        if request_binding != "text_format":
            raise ValueError(
                f"{profile_path}: 'output_contract.schema.request_binding' must be 'text_format'"
            )

        if not isinstance(module_name, str) or not module_name.strip():
            raise ValueError(
                f"{profile_path}: 'output_contract.schema.module' must be a non-empty string"
            )

        if not isinstance(object_name, str) or not object_name.strip():
            raise ValueError(
                f"{profile_path}: 'output_contract.schema.name' must be a non-empty string"
            )

        try:
            module = importlib.import_module(module_name.strip())
        except Exception as exc:
            raise ImportError(
                f"{profile_path}: failed to import module '{module_name}'"
            ) from exc

        try:
            contract = getattr(module, object_name.strip())
        except AttributeError as exc:
            raise ImportError(
                f"{profile_path}: module '{module_name}' has no attribute '{object_name}'"
            ) from exc

        return contract

    @staticmethod
    def _find_semantic_reference_example(
        examples: list[Any], profile_path: Path
    ) -> dict[str, Any] | None:
        """
        Find the example with:
        - purpose == semantic_reference
        - request_binding == instructions
        """
        matches: list[dict[str, Any]] = []

        for item in examples:
            if not isinstance(item, dict):
                raise ValueError(
                    f"{profile_path}: every item in 'output_contract.examples' must be a dict"
                )

            if (
                item.get("purpose") == "semantic_reference"
                and item.get("request_binding") == "instructions"
            ):
                matches.append(item)

        if len(matches) > 1:
            raise ValueError(
                f"{profile_path}: multiple semantic_reference examples with request_binding='instructions'"
            )

        return matches[0] if matches else None

    @staticmethod
    def _normalize_actions(value: Any) -> list[str]:
        """
        Validate actions field as list[str].
        """
        if value is None:
            return []

        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            raise ValueError("actions must be list[str]")

        return value
