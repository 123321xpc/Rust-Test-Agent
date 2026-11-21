_is_test_path(){
    local raw_target="$1"
    if [ -z "$raw_target" ]; then
        return 1
    fi
    local target
    target=$(realpath -m "$raw_target" 2>/dev/null || realpath "$raw_target" 2>/dev/null || printf '%s' "$raw_target")
    target="${target//\\//}"
    if [ -n "${ROOT:-}" ]; then
        local root_real
        root_real=$(realpath -m "$ROOT" 2>/dev/null || realpath "$ROOT" 2>/dev/null || printf '%s' "$ROOT")
        root_real="${root_real//\\//}"
        case "$target" in
            "$root_real"/*)
                target="${target#"$root_real"/}"
                ;;
            /tmp/*)
                return 1
                ;;
        esac
    else
        case "$target" in
            /tmp/*)
                return 1
                ;;
        esac
    fi
    while [[ "$target" == ./* ]]; do
        target="${target#./}"
    done
    local base="${target##*/}"
    case "$target" in
        */tests/*|tests/*|*/tests|tests|*/test/*|test/*|*/test|test|*/test_data/*|test_data/*|*/test_data|test_data)
            return 0
            ;;
    esac
    case "$base" in
        test.rs|tests.rs|test_*.rs|*_test.rs|*_tests.rs)
            return 0
            ;;
    esac
    return 1
}

_revert_test_files(){
    local target_root="${ROOT:-.}"
    local path_line
    while IFS= read -r path_line; do
        [ -z "$path_line" ] && continue
        local status="${path_line:0:2}"
        local rel_path="${path_line:3}"
        if [[ "$rel_path" == *" -> "* ]]; then
            local old_path="${rel_path%% -> *}"
            local new_path="${rel_path##* -> }"
            old_path="${old_path#\"}"
            old_path="${old_path%\"}"
            new_path="${new_path#\"}"
            new_path="${new_path%\"}"
            if _is_test_path "$target_root/$old_path" || _is_test_path "$target_root/$new_path"; then
                git -C "$target_root" reset -q HEAD -- "$old_path" "$new_path" 2>/dev/null || true
                git -C "$target_root" checkout -q -- "$old_path" 2>/dev/null || true
                if [ "$old_path" != "$new_path" ] && _is_test_path "$target_root/$new_path"; then
                    rm -rf -- "$target_root/$new_path" 2>/dev/null || true
                fi
            fi
            continue
        fi
        rel_path="${rel_path#\"}"
        rel_path="${rel_path%\"}"
        if _is_test_path "$target_root/$rel_path"; then
            if [[ "$status" == "??" ]]; then
                rm -rf -- "$target_root/$rel_path" 2>/dev/null || true
            else
                git -C "$target_root" reset -q HEAD -- "$rel_path" 2>/dev/null || true
                git -C "$target_root" checkout -q -- "$rel_path" 2>/dev/null || true
            fi
        fi
    done < <(git -C "$target_root" -c core.quotePath=false status --porcelain --untracked-files=all 2>/dev/null)
}

_print() {
    local total_lines=$(awk 'END {print NR}' $CURRENT_FILE)
    echo "[File: $(realpath $CURRENT_FILE) ($total_lines lines total)]"
    lines_above=$(jq -n "$CURRENT_LINE - $WINDOW/2" | jq '[0, .] | max | floor')
    lines_below=$(jq -n "$total_lines - $CURRENT_LINE - $WINDOW/2" | jq '[0, .] | max | floor')
    if [ $lines_above -gt 0 ]; then
        echo "($lines_above more lines above)"
    fi
    cat $CURRENT_FILE | grep -n $ | head -n $(jq -n "[$CURRENT_LINE + $WINDOW/2, $WINDOW/2] | max | floor") | tail -n $(jq -n "$WINDOW")
    if [ $lines_below -gt 0 ]; then
        echo "($lines_below more lines below)"
    fi
}

_rm_binaries(){
    for file in $(git status --porcelain | grep -E "^(M| M|\?\?|A| A)" | cut -c4-); do
        if [ -f "$file" ] && (file "$file" | grep -q "executable" || git check-attr binary "$file" | grep -q "binary: set"); then
            git rm -f "$file" 2>/dev/null || rm -f "$file"
            echo "Removed: $file"
        fi
    done
}

_constrain_line() {
    if [ -z "$CURRENT_FILE" ]
    then
        echo "No file open. Use the open command first."
        return
    fi
    local max_line=$(awk 'END {print NR}' $CURRENT_FILE)
    local half_window=$(jq -n "$WINDOW/2" | jq 'floor')
    export CURRENT_LINE=$(jq -n "[$CURRENT_LINE, $max_line - $half_window] | min")
    export CURRENT_LINE=$(jq -n "[$CURRENT_LINE, $half_window] | max")
}

# @yaml
# signature: open <path> [<line_number>]
# docstring: opens the file at the given path in the editor. If line_number is provided, the window will be move to include that line
# arguments:
#   path:
#     type: string
#     description: the path to the file to open
#     required: true
#   line_number:
#     type: integer
#     description: the line number to move the window to (if not provided, the window will start at the top of the file)
#     required: false
open() {
    if [ -z "$1" ]
    then
        echo "Usage: open <file>"
        return
    fi
    # Check if the second argument is provided
    if [ -n "$2" ]; then
        # Check if the provided argument is a valid number
        if ! [[ $2 =~ ^[0-9]+$ ]]; then
            echo "Usage: open <file> [<line_number>]"
            echo "Error: <line_number> must be a number"
            return  # Exit if the line number is not valid
        fi
        local max_line=$(awk 'END {print NR}' $1)
        if [ $2 -gt $max_line ]; then
            echo "Warning: <line_number> ($2) is greater than the number of lines in the file ($max_line)"
            echo "Warning: Setting <line_number> to $max_line"
            local line_number=$(jq -n "$max_line")  # Set line number to max if greater than max
        elif [ $2 -lt 1 ]; then
            echo "Warning: <line_number> ($2) is less than 1"
            echo "Warning: Setting <line_number> to 1"
            local line_number=$(jq -n "1")  # Set line number to 1 if less than 1
        else
            local OFFSET=$(jq -n "$WINDOW/6" | jq 'floor')
            local line_number=$(jq -n "[$2 + $WINDOW/2 - $OFFSET, 1] | max | floor")
        fi
    else
        local line_number=$(jq -n "$WINDOW/2")  # Set default line number if not provided
    fi

    if [ -f "$1" ]; then
        export CURRENT_FILE=$(realpath $1)
        export CURRENT_LINE=$line_number
        _constrain_line
        _print
    elif [ -d "$1" ]; then
        echo "Error: $1 is a directory. You can only open files. Use cd or ls to navigate directories."
    else
        echo "File $1 not found"
    fi
}

# @yaml
# signature: goto <line_number>
# docstring: moves the window to show <line_number>
# arguments:
#   line_number:
#     type: integer
#     description: the line number to move the window to
#     required: true
goto() {
    if [ $# -gt 1 ]; then
        echo "goto allows only one line number at a time."
        return
    fi
    if [ -z "$CURRENT_FILE" ]
    then
        echo "No file open. Use the open command first."
        return
    fi
    if [ -z "$1" ]
    then
        echo "Usage: goto <line>"
        return
    fi
    if ! [[ $1 =~ ^[0-9]+$ ]]
    then
        echo "Usage: goto <line>"
        echo "Error: <line> must be a number"
        return
    fi
    local max_line=$(awk 'END {print NR}' $CURRENT_FILE)
    if [ $1 -gt $max_line ]
    then
        echo "Error: <line> must be less than or equal to $max_line"
        return
    fi
    local OFFSET=$(jq -n "$WINDOW/6" | jq 'floor')
    export CURRENT_LINE=$(jq -n "[$1 + $WINDOW/2 - $OFFSET, 1] | max | floor")
    _constrain_line
    _print
}

# @yaml
# signature: scroll_down
# docstring: moves the window down {WINDOW} lines
scroll_down() {
    if [ -z "$CURRENT_FILE" ]
    then
        echo "No file open. Use the open command first."
        return
    fi
    export CURRENT_LINE=$(jq -n "$CURRENT_LINE + $WINDOW - $OVERLAP")
    _constrain_line
    _print
}

# @yaml
# signature: scroll_up
# docstring: moves the window up {WINDOW} lines
scroll_up() {
    if [ -z "$CURRENT_FILE" ]
    then
        echo "No file open. Use the open command first."
        return
    fi
    export CURRENT_LINE=$(jq -n "$CURRENT_LINE - $WINDOW + $OVERLAP")
    _constrain_line
    _print
}

# @yaml
# signature: create <filename>
# docstring: creates and opens a new file with the given name
# arguments:
#   filename:
#     type: string
#     description: the name of the file to create
#     required: true
create() {
    if [ -z "$1" ]; then
        echo "Usage: create <filename>"
        return
    fi
    if _is_test_path "$1"; then
        echo "Error: Creating test files is prohibited ($1). Only modify implementation source files or write temporary scripts in /tmp."
        return 1
    fi

    # Check if the file already exists
    if [ -e "$1" ]; then
        echo "Error: File '$1' already exists."
		open "$1"
        return
    fi

    # Create the file an empty new line
    printf "\n" > "$1"
    # Use the existing open command to open the created file
    open "$1"
}

# @yaml
# signature: submit
# docstring: submits your current code and terminates the session
submit() {
    cd $ROOT

    # Check if the patch file exists and is non-empty
    if [ -s "/root/test.patch" ]; then
        # Apply the patch in reverse
        git apply -R < "/root/test.patch"
    fi

    _revert_test_files
    git add -A
    # rm binaries files
    _rm_binaries
    git diff --cached > model.patch
    echo "<<SUBMISSION||"
    cat model.patch
    echo "||SUBMISSION>>"
}
