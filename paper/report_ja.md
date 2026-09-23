# Mean-Field Capacity Control：日本語報告

## 概要

公開Azure Functionsトレースから得たワークロード形状を用い、有限$N$ JSQ($d$)シミュレーションと準定常mean-field MPCを評価した。結果はAzure本番デプロイではなく、制御されたシミュレーションである。

## 主結果

high-volumeにおけるEWMA MF-MPC対reactiveの厳密なmatched-budget比較では、p95待ち時間を56.7%削減した（95% paired bootstrap CI [52.8%, 60.6%]）。Matching status was determined on validation only; reported resource fractions below are test-set means. つまり、マッチング判定はvalidationだけで決定し、報告するresource fractionはtest-set平均である。EWMAとreactiveのtest-set平均resource fractionは0.274と0.273、相対差は0.064%で、設定した2%以内である。

GP-UCBはhigh-volumeでreactive比94.4%のp95削減を示すが、resource fractionは0.758対0.273であり、資源未一致の記述的結果である。同一コスト優位性や本番優位性は主張しない。

burstyではGP-UCBはGP-meanに対してp95待ち時間に明確な差を示さなかった。paired percent reductionは-2.5%（95% CI [-14.8%, 8.9%]）で、区間はゼロを含む。

## 方法上の注意

シナリオ適格性は、以前の退化したheld-out分割を発見した後、controller再評価前に固定した。学習・validation・testを時系列分割し、GP fittingは学習、beta校正とresource matchingはvalidationだけで行い、testは設定固定後の評価にのみ使用した。詳細な数式・図・再現手順は英語PDFと`docs/RESEARCH_PROTOCOL.md`に記載する。
