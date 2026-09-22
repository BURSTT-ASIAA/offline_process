#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOF'
Usage: auto_inten_rebin.sh TARGET DATE ZMIN ZMAX ROWS [HOURS]

	TARGET  source name understood by obs_planner.py (for example, Crab or Sun)
	DATE    local date (UTC+8) in YYMMDD or YYYYMMDD format
  ZMIN    lower color-scale limit
  ZMAX    upper color-scale limit
  ROWS    rows passed to plot_bf256_16k.py, for example 7 or "0 2 4"
	HOURS   half-width around UTC transit in hours (default: 1.5)

Intensity filename timestamps are interpreted as Fushan local time (UTC+8).
EOF
}

if [[ $# -lt 5 || $# -gt 6 ]]; then
	usage >&2
	exit 2
fi

TARGET=$1
DATE=$2
ZMIN=$3
ZMAX=$4
ROWS=$5
HOURS=${6:-1.5}

TARGET_LOWER=${TARGET,,}
case $TARGET_LOWER in
	crab|taua)
		SOURCE_DIR=crab
		;;
	*)
		SOURCE_DIR=$TARGET_LOWER
		;;
esac

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXECUTION_DIR=$PWD
FINDER=$SCRIPT_DIR/find_near_transit.py
PLOTTER=/data/kylin/bin/plot_bf256_16k.py
DATE_TOKEN=${DATE//-/}
DATE_TOKEN=${DATE_TOKEN: -6}
OUTDIR=$EXECUTION_DIR/$DATE_TOKEN/$SOURCE_DIR

mkdir -p "$OUTDIR"
linked=0
while IFS=$'\t' read -r _ _ source_file; do
	[[ -n $source_file ]] || continue
	host=$(basename "$(dirname "$(dirname "$(dirname "$source_file")")")")
	beam_dir=${host/burstt/b}
	mkdir -p "$OUTDIR/$beam_dir"
	ln -sfn "$source_file" "$OUTDIR/$beam_dir/$(basename "$source_file")"
	linked=$((linked + 1))
done < <(python "$FINDER" --target "$TARGET" --date "$DATE" --hours "$HOURS")

if [[ $linked -eq 0 ]]; then
	echo "No intensity files found near ${TARGET} transit on ${DATE}." >&2
	exit 1
fi

cd "$OUTDIR"
echo "Linked $linked files into $OUTDIR"
echo "Plotting with zlim $ZMIN $ZMAX and rows '$ROWS'"

plot_args=(
	--flim 300 700
	--sum 3200
	--ochan 128
	--fwin 4
	--zlim "$ZMIN" "$ZMAX"
	--rows "$ROWS"
	-v
)

plot_ring() {
	local ring=$1
	local beam_dir=$2
	local status=0
	python "$PLOTTER" -r "$ring" "$beam_dir" "${plot_args[@]}" || status=$?
	if [[ $status -ne 0 && $status -ne 1 ]]; then
		return "$status"
	fi
}

for ring_dir in 15 16; do
	beam_dir=b$ring_dir
	echo $beam_dir
	[[ -d $beam_dir && -n $(find "$beam_dir" -type l -print -quit) ]] || continue
	case $ring_dir in
		16) ring=0 ;;
		15) ring=2 ;;
		12) ring=4 ;;
		11) ring=6 ;;
	esac
	echo $beam_dir $ring
	plot_ring "$ring" "$beam_dir"
	if [[ $ring_dir == 15 ]]; then
		plot_ring 0 "$beam_dir"
	fi
done

