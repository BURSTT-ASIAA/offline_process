#!/usr/bin/perl

$ldir = 'logs';
`mkdir -p $ldir`;

$block0 = 1000;
$totblock = 1500;
$njob = 6;

$stepblock = int($totblock / $njob);


for $j (1 .. 1) {
	$block0 += $j*$totblock;
	for $i (0 .. $njob-1) {
		$start = $block0 + $i*$stepblock;
		$cmd = "./stream_search_delay2.py $start -l $stepblock";
		$lfile = sprintf("$ldir/search$j-%02d.log", $i);
		`$cmd > $lfile 2>&1 &`;
	};
};
