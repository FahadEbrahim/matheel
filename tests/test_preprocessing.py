import pytest

from matheel.preprocessing import available_preprocess_languages, available_preprocess_modes, preprocess_code


BASIC_LANGUAGE_CASES = (
    (
        "java",
        """
        int value = 1; // trailing note
        /* remove this block */
        int second = 2;
        """,
        "int value = 1; int second = 2;",
    ),
    (
        "python",
        """
        value = 1  # trailing note
        # remove this line
        second = 2
        """,
        "value = 1 second = 2",
    ),
    (
        "c",
        """
        int value = 1; // trailing note
        /* remove this block */
        int second = 2;
        """,
        "int value = 1; int second = 2;",
    ),
    (
        "cpp",
        """
        int value = 1; // trailing note
        /* remove this block */
        int second = 2;
        """,
        "int value = 1; int second = 2;",
    ),
    (
        "go",
        """
        value := 1 // trailing note
        /* remove this block */
        second := 2
        """,
        "value := 1 second := 2",
    ),
    (
        "javascript",
        """
        let value = 1; // trailing note
        /* remove this block */
        let second = 2;
        """,
        "let value = 1; let second = 2;",
    ),
    (
        "typescript",
        """
        let value: number = 1; // trailing note
        /* remove this block */
        let second: number = 2;
        """,
        "let value: number = 1; let second: number = 2;",
    ),
    (
        "kotlin",
        """
        val value: Int = 1 // trailing note
        /* remove this block */
        val second: Int = 2
        """,
        "val value: Int = 1 val second: Int = 2",
    ),
    (
        "scala",
        """
        val value: Int = 1 // trailing note
        /* remove this block */
        val second: Int = 2
        """,
        "val value: Int = 1 val second: Int = 2",
    ),
    (
        "swift",
        """
        let value: Int = 1 // trailing note
        /* remove this block */
        let second: Int = 2
        """,
        "let value: Int = 1 let second: Int = 2",
    ),
    (
        "solidity",
        """
        uint256 value = 1; // trailing note
        /* remove this block */
        uint256 second = 2;
        """,
        "uint256 value = 1; uint256 second = 2;",
    ),
    (
        "dart",
        """
        final int value = 1; // trailing note
        /* remove this block */
        final int second = 2;
        """,
        "final int value = 1; final int second = 2;",
    ),
    (
        "php",
        """
        <?php
        $value = 1; // trailing note
        # remove this line
        $second = 2;
        """,
        "<?php $value = 1; $second = 2;",
    ),
    (
        "ruby",
        """
        value = 1 # trailing note
        # remove this line
        second = 2
        """,
        "value = 1 second = 2",
    ),
    (
        "rust",
        """
        let value = 1; // trailing note
        /* remove this block */
        let second = 2;
        """,
        "let value = 1; let second = 2;",
    ),
    (
        "csharp",
        """
        int value = 1; // trailing note
        /* remove this block */
        int second = 2;
        """,
        "int value = 1; int second = 2;",
    ),
    (
        "lua",
        """
        local value = 1 -- trailing note
        --[[ remove this block ]]
        local second = 2
        """,
        "local value = 1 local second = 2",
    ),
    (
        "julia",
        """
        value = 1 # trailing note
        # remove this line
        second = 2
        """,
        "value = 1 second = 2",
    ),
    (
        "r",
        """
        value <- 1 # trailing note
        # remove this line
        second <- 2
        """,
        "value <- 1 second <- 2",
    ),
    (
        "objc",
        """
        int value = 1; // trailing note
        /* remove this block */
        int second = 2;
        """,
        "int value = 1; int second = 2;",
    ),
)

ADVANCED_LANGUAGE_CASES = (
    (
        "java",
        """
        package demo;
        import java.util.List;
        class Demo {
            static int AddValue(int totalCount, int item2) {
                String text = "hello";
                int n = 42;
                return totalCount + item2 + n;
            }
        }
        """,
        ("package demo", "import java.util.List", "AddValue", "totalCount", "item2"),
    ),
    (
        "python",
        """
        import os
        from math import sqrt
        def AddValue(total_count, item2):
            text = "hello"
            n = 42
            return total_count + item2 + n
        """,
        ("import os", "from math import sqrt", "AddValue", "total_count", "item2"),
    ),
    (
        "c",
        """
        #include <stdio.h>
        int AddValue(int totalCount, int item2) {
            char* text = "hello";
            int n = 42;
            return totalCount + item2 + n;
        }
        """,
        ("#include", "AddValue", "totalCount", "item2"),
    ),
    (
        "cpp",
        """
        #include <vector>
        using namespace std;
        int AddValue(int totalCount, int item2) {
            auto text = "hello";
            int n = 42;
            return totalCount + item2 + n;
        }
        """,
        ("#include", "using namespace std", "AddValue", "totalCount", "item2"),
    ),
    (
        "go",
        """
        package main
        import (
            "fmt"
            alias "os"
        )
        func AddValue(totalCount int, item2 int) int {
            text := "hello"
            n := 42
            return totalCount + item2 + n
        }
        """,
        ("package main", 'import (', '"fmt"', '"os"', "AddValue", "totalCount", "item2"),
    ),
    (
        "javascript",
        """
        import { join } from "path";
        const fs = require("fs");
        function AddValue(totalCount, item2) {
            const text = "hello";
            const n = 42;
            return totalCount + item2 + n;
        }
        """,
        ('import { join } from "path"', 'const fs = require("fs")', "AddValue", "totalCount", "item2"),
    ),
    (
        "typescript",
        """
        import type { Options } from "./types";
        const fs = require("fs");
        function AddValue(totalCount: number, item2: number): number {
            const text: string = "hello";
            const n: number = 42;
            return totalCount + item2 + n;
        }
        """,
        ('import type { Options } from "./types"', 'const fs = require("fs")', "AddValue", "totalCount", "item2"),
    ),
    (
        "kotlin",
        """
        package demo
        import kotlin.math.abs
        fun AddValue(totalCount: Int, item2: Int): Int {
            val text = "hello"
            val n = 42
            return totalCount + item2 + n
        }
        """,
        ("package demo", "import kotlin.math.abs", "AddValue", "totalCount", "item2"),
    ),
    (
        "scala",
        """
        package demo
        import scala.math.abs
        def AddValue(totalCount: Int, item2: Int): Int = {
          val text = "hello"
          val n = 42
          return totalCount + item2 + n
        }
        """,
        ("package demo", "import scala.math.abs", "AddValue", "totalCount", "item2"),
    ),
    (
        "swift",
        """
        import Foundation
        func AddValue(totalCount: Int, item2: Int) -> Int {
            let text = "hello"
            let n = 42
            return totalCount + item2 + n
        }
        """,
        ("import Foundation", "AddValue", "totalCount", "item2"),
    ),
    (
        "solidity",
        """
        pragma solidity ^0.8.0;
        import "./Math.sol";
        contract Demo {
            function AddValue(uint256 totalCount, uint256 item2) public pure returns (uint256) {
                string memory text = "hello";
                uint256 n = 42;
                return totalCount + item2 + n;
            }
        }
        """,
        ("pragma solidity", 'import "./Math.sol"', "AddValue", "totalCount", "item2"),
    ),
    (
        "dart",
        """
        import "dart:math";
        int AddValue(int totalCount, int item2) {
          final text = "hello";
          final n = 42;
          return totalCount + item2 + n;
        }
        """,
        ('import "dart:math"', "AddValue", "totalCount", "item2"),
    ),
    (
        "php",
        """
        <?php
        use Foo\\Bar;
        require_once "lib.php";
        function AddValue($totalCount, $item2) {
            $text = "hello";
            $n = 42;
            return $totalCount + $item2 + $n;
        }
        """,
        (r"use Foo\Bar", 'require_once "lib.php"', "AddValue", "totalCount", "item2"),
    ),
    (
        "ruby",
        """
        require "json"
        require_relative "helper"
        def AddValue(total_count, item2)
          text = "hello"
          n = 42
          return total_count + item2 + n
        end
        """,
        ('require "json"', 'require_relative "helper"', "AddValue", "total_count", "item2"),
    ),
    (
        "rust",
        """
        use std::fmt;
        fn add_value(total_count: i32, item2: i32) -> i32 {
            let text = "hello";
            let n = 42;
            return total_count + item2 + n;
        }
        """,
        ("use std::fmt", "total_count", "item2"),
    ),
    (
        "csharp",
        """
        global using System;
        using static System.Math;
        class Demo {
            static int AddValue(int totalCount, int item2) {
                var text = "hello";
                var n = 42;
                return totalCount + item2 + n;
            }
        }
        """,
        ("global using System", "using static System.Math", "AddValue", "totalCount", "item2"),
    ),
    (
        "lua",
        """
        require "json"
        local function AddValue(total_count, item2)
            local text = "hello"
            local n = 42
            return total_count + item2 + n
        end
        """,
        ('require "json"', "AddValue", "total_count", "item2"),
    ),
    (
        "julia",
        """
        using Statistics
        import Base:+
        function AddValue(total_count, item2)
            text = "hello"
            n = 42
            return total_count + item2 + n
        end
        """,
        ("using Statistics", "import Base:+", "AddValue", "total_count", "item2"),
    ),
    (
        "r",
        """
        library(dplyr)
        source("helpers.R")
        AddValue <- function(total_count, item2) {
          text <- "hello"
          n <- 42
          return(total_count + item2 + n)
        }
        """,
        ("library(dplyr)", 'source("helpers.R")', "AddValue", "total_count", "item2"),
    ),
    (
        "objc",
        """
        #import <Foundation/Foundation.h>
        @implementation Demo
        - (int)AddValue:(int)totalCount item2:(int)item2 {
            char* text = "hello";
            int n = 42;
            return totalCount + item2 + n;
        }
        @end
        """,
        ("#import <Foundation/Foundation.h>", "AddValue", "totalCount", "item2"),
    ),
)


def test_basic_preprocess_removes_comments_and_collapses_whitespace():
    code = """
    int value = 1; // trailing note
    /* remove this block */
    # comment
    int second = 2;
    """

    assert preprocess_code(code, mode="basic") == "int value = 1; int second = 2;"


@pytest.mark.parametrize("language,code,expected", BASIC_LANGUAGE_CASES)
def test_basic_preprocess_supports_scoped_languages(language, code, expected):
    processed = preprocess_code(code, mode="basic")

    assert processed == expected


def test_basic_preprocess_keeps_cpp_directives():
    code = """
    #include <stdio.h>
    int value = 1; # inline comment
    """

    assert preprocess_code(code, mode="basic") == "#include <stdio.h> int value = 1;"


def test_basic_preprocess_keeps_csharp_hash_directives():
    code = """
    #region Demo
    int value = 1; # inline comment
    #endregion
    """

    assert preprocess_code(code, mode="basic") == "#region Demo int value = 1; #endregion"


def test_available_preprocess_modes_expose_advanced_only():
    modes = available_preprocess_modes()

    assert "advanced" in modes
    assert "synsem_basic" not in modes


def test_available_preprocess_languages_match_regression_scope():
    assert available_preprocess_languages() == (
        "java",
        "python",
        "c",
        "cpp",
        "go",
        "javascript",
        "typescript",
        "kotlin",
        "scala",
        "swift",
        "solidity",
        "dart",
        "php",
        "ruby",
        "rust",
        "csharp",
        "lua",
        "julia",
        "r",
        "objc",
    )


def test_normalize_preprocess_drops_blank_lines_only():
    code = "line_a  \n\n\nline_b\n"

    assert preprocess_code(code, mode="normalize") == "line_a\nline_b"


@pytest.mark.parametrize("language,code,blocked_phrases", ADVANCED_LANGUAGE_CASES)
def test_advanced_preprocess_supports_scoped_languages(language, code, blocked_phrases):
    processed = preprocess_code(code, mode="advanced")

    for phrase in blocked_phrases:
        assert phrase not in processed
    assert "<STR>" in processed
    assert "<NUM>" in processed
    assert "return" in processed
    assert "id1" in processed


def test_advanced_preprocess_strips_imports_literals_and_identifiers():
    code = """
    import os
    from math import sqrt
    #include <stdio.h>
    using namespace std;
    def AddValue(total_count, item2):
        text = "hello"
        n = 42
        return total_count + item2 + n
    """

    processed = preprocess_code(code, mode="advanced")

    assert "import os" not in processed
    assert "from math import sqrt" not in processed
    assert "#include" not in processed
    assert "using namespace" not in processed
    assert "<STR>" in processed
    assert "<NUM>" in processed
    assert "AddValue" not in processed
    assert "total_count" not in processed
    assert "item2" not in processed
    assert "id1" in processed


def test_preprocess_code_rejects_unknown_mode():
    with pytest.raises(ValueError):
        preprocess_code("x = 1", mode="unsupported-mode")
