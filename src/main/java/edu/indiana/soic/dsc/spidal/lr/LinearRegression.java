package edu.indiana.soic.dsc.spidal.lr;

import java.util.*;
import java.io.*;



public class LinearRegression {
    public static void main(String[] args) throws Exception {


        int n= 1000;
        double []x = new double[n];
        double []y = new double[n];

        String filename = "/Users/madhu/Desktop/regcopy.txt";
        FileReader fr = new FileReader(filename);
        BufferedReader br = new BufferedReader(fr);

        String line;
        String[] split;
        int i = 0;

        while((line=br.readLine()) != null){
            split = line.split("\\s+");
            x[i] = Double.parseDouble((split[0]));
            y[i] = Double.parseDouble(split[1]);
            i++;
            //System.out.print(x[i] + "  ");
            //System.out.print(y[i] + "  ");
        }


        double sumx = 0.0; double sumy = 0.0; double sumz=0.0;
        for (i=0; i<n; i++) {
            sumx  = sumx +  x[i];
            sumz = sumz + (x[i] * x[i]);
            sumy  = sumy + y[i];

        }

        double xAvg = sumx / n;
        double yAvg = sumy / n;
        double xX = 0.0; double yY = 0.0; double xY = 0.0;
        for ( i = 0; i < n; i++) {
            xX += (x[i] - xAvg) * (x[i] - xAvg);
            yY += (y[i] - yAvg) * (y[i] - yAvg);
            xY += (x[i] - xAvg) * (y[i] - yAvg);

        }

        double beta1 = xY / xX;
        double beta0 = yAvg - beta1 * xAvg;

        System.out.println("y   = " + beta1 + " * x + " + beta0);


        int df = n - 2;
        double rss = 0.0;
        double ssr = 0.0;

        for ( i = 0; i < n; i++) {

            double fit = beta1*x[i] + beta0;
            rss += (fit - y[i]) * (fit - y[i]);
            ssr += (fit - yAvg) * (fit - yAvg);

        }

        double R2    = ssr / yY;
        double svar  = rss / df;
        double svar1 = svar / xX;
        double svar0 = svar/n + xAvg*xAvg*svar1;
        System.out.println("R^2                 = " + R2);
        System.out.println("STD error of beta_1 = " + Math.sqrt(svar1));
        svar0 = svar * sumz / (n * xX);
        System.out.println("STD error of beta_0 = " + Math.sqrt(svar0));
        System.out.println("SSTO = " + yY);
        System.out.println("SSE  = " + rss);
        System.out.println("SSR  = " + ssr);
    }


}

