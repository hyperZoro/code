// A UI Elements
const elsA = {
    amount: document.getElementById('loanAmountA'),
    rate: document.getElementById('interestRateA'),
    term: document.getElementById('loanTermA'),
    overpay: document.getElementById('overpaymentA')
};
const outA = {
    mTotal: document.getElementById('mTotalA'),
    mPrinc: document.getElementById('mPrincA'),
    mInt: document.getElementById('mIntA'),
    yTotal: document.getElementById('yTotalA'),
    yPrinc: document.getElementById('yPrincA'),
    yInt: document.getElementById('yIntA'),
    payoff: document.getElementById('payoffTimeA'),
    totInt: document.getElementById('totalIntA')
};

// B UI Elements
const elsB = {
    amount: document.getElementById('loanAmountB'),
    rate: document.getElementById('interestRateB'),
    term: document.getElementById('loanTermB'),
    overpay: document.getElementById('overpaymentB')
};
const outB = {
    mTotal: document.getElementById('mTotalB'),
    mPrinc: document.getElementById('mPrincB'),
    mInt: document.getElementById('mIntB'),
    yTotal: document.getElementById('yTotalB'),
    yPrinc: document.getElementById('yPrincB'),
    yInt: document.getElementById('yIntB'),
    payoff: document.getElementById('payoffTimeB'),
    totInt: document.getElementById('totalIntB')
};

let myChart;

const formatter = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0
});

function getValues(els) {
    return {
        P: parseFloat(els.amount.value) || 0,
        rAnn: parseFloat(els.rate.value) || 0,
        years: parseFloat(els.term.value) || 0,
        overpayment: parseFloat(els.overpay.value) || 0
    };
}

function calcScenario(params) {
    const data = {
        yearlyPrincipal: [],
        yearlyInterest: [],
        firstMonth: { payment: 0, principal: 0, interest: 0 },
        year1: { payment: 0, principal: 0, interest: 0 },
        payoff: { years: 0, months: 0 },
        totalInterest: 0,
        valid: false
    };

    const { P, rAnn, years, overpayment } = params;
    if (P <= 0 || rAnn <= 0 || years <= 0) return data;

    data.valid = true;
    const r = rAnn / 100 / 12;
    const n = years * 12;

    const M = P * (r * Math.pow(1 + r, n)) / (Math.pow(1 + r, n) - 1);
    const actualMonthlyPayment = M + overpayment;

    let balance = P;
    let months = 0;
    
    let currentYearPrincipal = 0;
    let currentYearInterest = 0;
    let totalInterest = 0;

    while (balance > 0.01 && months < 1200) {
        const interest = balance * r;
        let principal = actualMonthlyPayment - interest;
        
        if (principal > balance) {
            principal = balance;
            balance = 0;
        } else {
            balance -= principal;
        }

        if (months === 0) {
            data.firstMonth = {
                payment: principal + interest,
                principal: principal,
                interest: interest
            };
        }

        currentYearPrincipal += principal;
        currentYearInterest += interest;
        totalInterest += interest;
        
        months++;

        if (months % 12 === 0 || balance <= 0) {
            data.yearlyPrincipal.push(currentYearPrincipal);
            data.yearlyInterest.push(currentYearInterest);
            
            if (months <= 12) {
                 data.year1 = { 
                     payment: currentYearPrincipal + currentYearInterest,
                     principal: currentYearPrincipal, 
                     interest: currentYearInterest 
                 };
            }

            currentYearPrincipal = 0;
            currentYearInterest = 0;
        }
    }

    // fallback for year 1 if loan is somehow paid off in under 12 months immediately
    if (months < 12 && data.yearlyPrincipal.length > 0) {
        data.year1 = { 
             payment: data.yearlyPrincipal[0] + data.yearlyInterest[0],
             principal: data.yearlyPrincipal[0], 
             interest: data.yearlyInterest[0] 
        };
    }

    data.totalInterest = totalInterest;
    data.payoff.years = Math.floor(months / 12);
    data.payoff.months = months % 12;

    return data;
}

function updateDOM(out, data) {
    if (!data.valid) {
        out.mTotal.innerText = '$0';
        out.mPrinc.innerText = '$0';
        out.mInt.innerText = '$0';
        out.yTotal.innerText = '$0';
        out.yPrinc.innerText = '$0';
        out.yInt.innerText = '$0';
        out.payoff.innerText = '0 Yrs';
        out.totInt.innerText = '$0';
        return;
    }

    out.mTotal.innerText = formatter.format(data.firstMonth.payment);
    out.mPrinc.innerText = formatter.format(data.firstMonth.principal);
    out.mInt.innerText = formatter.format(data.firstMonth.interest);

    out.yTotal.innerText = formatter.format(data.year1.payment);
    out.yPrinc.innerText = formatter.format(data.year1.principal);
    out.yInt.innerText = formatter.format(data.year1.interest);

    let payoffText = '';
    if (data.payoff.years > 0) payoffText += `${data.payoff.years} Yrs `;
    if (data.payoff.months > 0) payoffText += `${data.payoff.months} Mos`;
    if (payoffText === '') payoffText = '0 Mos';

    out.payoff.innerText = payoffText.trim();
    out.totInt.innerText = formatter.format(data.totalInterest);
}

function updateChart(dataA, dataB) {
    const lenA = dataA.valid ? dataA.yearlyPrincipal.length : 0;
    const lenB = dataB.valid ? dataB.yearlyPrincipal.length : 0;
    const maxLen = Math.max(lenA, lenB, 1);

    const labels = Array.from({length: maxLen}, (_, i) => `Yr ${i + 1}`);

    const pad = (arr, len) => {
        const res = [...arr];
        while (res.length < len) res.push(0);
        return res;
    };

    const dsAPrinc = pad(dataA.valid ? dataA.yearlyPrincipal : [], maxLen);
    const dsAInt = pad(dataA.valid ? dataA.yearlyInterest : [], maxLen);
    const dsBPrinc = pad(dataB.valid ? dataB.yearlyPrincipal : [], maxLen);
    const dsBInt = pad(dataB.valid ? dataB.yearlyInterest : [], maxLen);

    const ctx = document.getElementById('amortizationChart').getContext('2d');
    if (myChart) myChart.destroy();

    myChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'A: Principal',
                    data: dsAPrinc,
                    backgroundColor: 'rgba(59, 130, 246, 0.9)', 
                    stack: 'StackA',
                    barPercentage: 0.9,
                    categoryPercentage: 0.8
                },
                {
                    label: 'A: Interest',
                    data: dsAInt,
                    backgroundColor: 'rgba(59, 130, 246, 0.3)', 
                    stack: 'StackA'
                },
                {
                    label: 'B: Principal',
                    data: dsBPrinc,
                    backgroundColor: 'rgba(139, 92, 246, 0.9)',
                    stack: 'StackB',
                    barPercentage: 0.9,
                    categoryPercentage: 0.8
                },
                {
                    label: 'B: Interest',
                    data: dsBInt,
                    backgroundColor: 'rgba(139, 92, 246, 0.3)',
                    stack: 'StackB'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#e2e8f0',
                    bodyColor: '#e2e8f0',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            if (context.raw === 0 && context.datasetIndex > 1) return null; // hide zero for B if it doesn't exist
                            return context.dataset.label + ': ' + formatter.format(context.raw);
                        }
                    }
                }
            },
            scales: {
                x: {
                    stacked: true,
                    grid: { display: false },
                    ticks: { color: '#94a3b8', maxTicksLimit: 20 }
                },
                y: {
                    stacked: true,
                    grid: { color: 'rgba(255,255,255,0.05)', drawBorder: false },
                    ticks: { 
                        color: '#94a3b8',
                        callback: function(value) { return '$' + (value/1000) + 'k'; }
                    }
                }
            }
        }
    });
}

function calculateAll() {
    const paramsA = getValues(elsA);
    const paramsB = getValues(elsB);

    const dataA = calcScenario(paramsA);
    const dataB = calcScenario(paramsB);

    updateDOM(outA, dataA);
    updateDOM(outB, dataB);

    updateChart(dataA, dataB);
}

// Add event listeners
[...Object.values(elsA), ...Object.values(elsB)].forEach(el => {
    el.addEventListener('input', calculateAll);
});

// Init
calculateAll();
