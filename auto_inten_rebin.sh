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
	--subicb
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

# each beam dir holds two ring orders: ring0 files -> even order, ring1 files -> odd order
for ring in 0 1 2 3 4 5 6 7; do
	case $ring in
		0|1) beam_dir=b16 ;;
		2|3) beam_dir=b15 ;;
		4|5) beam_dir=b12 ;;
		6|7) beam_dir=b11 ;;
	esac
	ring_name=ring$((ring % 2))
	if [[ ! -d $beam_dir || -z $(find "$beam_dir" -name "intensity_${ring_name}_*" -print -quit) ]]; then
		echo "Skipping order $ring: no ${ring_name} files in $beam_dir" >&2
		continue
	fi
	echo "Plotting order $ring from $beam_dir ($ring_name)"
	plot_ring "$ring" "$beam_dir"
done

