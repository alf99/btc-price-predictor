import React, { useEffect, useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line } from 'react-chartjs-2';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const PriceChart = ({ data, interval }) => {
  const [chartData, setChartData] = useState({
    labels: [],
    datasets: []
  });
  
  const [chartOptions, setChartOptions] = useState({});
  
  useEffect(() => {
    if (!data || data.length === 0) return;
    
    // Format data based on interval
    let labels, prices;
    
    if (interval === '1h') {
      labels = data.map(item => {
        const date = new Date(item.timestamp);
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      });
      prices = data.map(item => item.close);
    } else {
      labels = data.map(item => {
        const date = new Date(item.timestamp);
        return date.toLocaleDateString();
      });
      prices = data.map(item => item.price);
    }
    
    // Create gradient for area under the line
    const ctx = document.createElement('canvas').getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(14, 165, 233, 0.5)');
    gradient.addColorStop(1, 'rgba(14, 165, 233, 0)');
    
    setChartData({
      labels,
      datasets: [
        {
          label: 'BTC Price',
          data: prices,
          borderColor: 'rgb(14, 165, 233)',
          backgroundColor: gradient,
          borderWidth: 2,
          pointBackgroundColor: 'rgb(14, 165, 233)',
          pointBorderColor: '#fff',
          pointBorderWidth: 1,
          pointRadius: 3,
          pointHoverRadius: 5,
          fill: true,
          tension: 0.4
        }
      ]
    });
    
    setChartOptions({
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          mode: 'index',
          intersect: false,
          callbacks: {
            label: function(context) {
              return `Price: $${context.raw.toLocaleString('en-US', {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
              })}`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: {
            display: false
          }
        },
        y: {
          grid: {
            color: 'rgba(0, 0, 0, 0.05)'
          },
          ticks: {
            callback: function(value) {
              return '$' + value.toLocaleString('en-US', {
                minimumFractionDigits: 0,
                maximumFractionDigits: 0
              });
            }
          }
        }
      },
      interaction: {
        mode: 'nearest',
        axis: 'x',
        intersect: false
      }
    });
  }, [data, interval]);
  
  if (!data || data.length === 0) {
    return (
      <div className="flex justify-center items-center h-64 bg-gray-50 rounded-lg">
        <p className="text-gray-500">No data available</p>
      </div>
    );
  }
  
  return (
    <div className="h-80">
      <Line data={chartData} options={chartOptions} />
    </div>
  );
};

export default PriceChart;