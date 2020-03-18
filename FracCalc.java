import java.util.*; 

public class FracCalc {

    public static void main(String[] args) 
    {
        Scanner inputScanner = new Scanner(System.in);
        String inputLine = inputScanner.nextLine();
        System.out.println(produceAnswer(inputLine));
    }
    
    //GCF helper function
    
    public static int gcd(int a1, int a2){
        int gcd = 1;
        for(int i = 1; i<=a1 && i<=a2; i++){
            if(a1%i==0 && a2%i==0){
            
                gcd = i;
            
            }
        }
        return gcd;
    
    }
    
    //LCM helper function
    
    public static int lcm(int a, int b){
        return a * (b / gcd(a, b));
    }
    
    //Trim helper because I like writing trim(num) versus num.trim(). Less typing.
    
    public static String trim(String num){
        return num.trim();
    }

    //parseNum function. Used to parse a specific index into whole number,
    //numer and denom values. Exceptions for whole numbers, improper fractions,
    //etc.
    
    public static int[][] parseNum(String[] info, int indexNum, int[][] fractions){
    
        if(info[indexNum].indexOf("_") != -1){
           //This is a mixed number!
           
           String whole = info[indexNum].substring(0,info[indexNum].indexOf("_"));
           String numer = info[indexNum].substring(info[indexNum].indexOf("_")+1, info[indexNum].indexOf("/"));
           String denom = info[indexNum].substring(info[indexNum].indexOf("/")+1);
           
           int wholeVal = Integer.parseInt(trim(whole));
           int numerVal = Integer.parseInt(trim(numer));
           int denomVal = Integer.parseInt(trim(denom));

           fractions[0][indexNum] = (denomVal*wholeVal)+numerVal;
           fractions[1][indexNum] = denomVal;
           
           if(wholeVal<0){
               fractions[2][indexNum] = -1;
               
               fractions[0][indexNum] = (denomVal*wholeVal) - numerVal;
               fractions[1][indexNum] = denomVal;
               
           }
           else{
               fractions[2][indexNum] = 1;
               
               fractions[0][indexNum] = (denomVal*wholeVal)+numerVal;
               fractions[1][indexNum] = denomVal;
           }
        }
        else if(info[indexNum].indexOf("/")==-1){
           //This is a whole number!
           
           String whole = info[indexNum];
           
           int wholeVal = Integer.parseInt(trim(whole));
           
           if(wholeVal < 0) fractions[2][indexNum] = -1;
           
           fractions[0][indexNum] = wholeVal;
           fractions[1][indexNum] = 1;
        }
        else{
           //This is a fraction!
           
           String numer = info[indexNum].substring(info[indexNum].indexOf("_")+1, info[indexNum].indexOf("/"));
           String denom = info[indexNum].substring(info[indexNum].indexOf("/")+1);
           
           int numerVal = Integer.parseInt(trim(numer));
           int denomVal;
           if(numerVal == 0) denomVal = 1;
           denomVal = Integer.parseInt(trim(denom));
           
           if(numerVal < 0 || denomVal < 0) fractions[2][indexNum] = -1;
           
           fractions[0][indexNum] = numerVal;
           fractions[1][indexNum] = denomVal;

        }
       
        return fractions;
    
    }
    
    //calculator function
    
    public static String produceAnswer(String input)
    { 
        
        //Input parser splits string by space into a three part array:
        //number 1, operator, number 2
        
        String[] info = new String[3];
        String wordTemp="";
        int k = 0;
        for(int j = 0; j<input.length(); j++){
            if(Character.toString(input.charAt(j)).equals(" ") && j!=0){
                 info[k] = wordTemp;
                 if(k==1){ info[k+1] = input.substring(j);}
                 k++;
                 wordTemp = "";
            }
            else{ wordTemp += Character.toString(input.charAt(j));}
        }
        
        //The below array contains the improper fraction forms of both nums,
        //plus a sign bit for pos/neg and a space for the result.
        
        int[][] fractions = new int[3][4];
        
        //Parsing both numbers into the fractions array
        
        fractions = parseNum(info, 0, fractions);
        fractions = parseNum(info, 2, fractions);
        
        //switch statement performs one of four operations (one for each
        //operator). Simplifications and exceptions are left out to prevent
        //code repitition.
        
        switch(info[1]){
        
           case "+":
           
                int newDenomVal = lcm(fractions[1][0], fractions[1][2]);
                fractions[0][0] *= newDenomVal/fractions[1][0];
                fractions[0][2] *= newDenomVal/fractions[1][2];
                int impnum = fractions[0][0] + fractions[0][2];
                
                fractions[0][3] = impnum;
                fractions[1][3] = newDenomVal;
                break;
                
                
           case "-":
           
                int newDenomVal2 = lcm(fractions[1][0], fractions[1][2]);
                fractions[0][0] *= newDenomVal2/fractions[1][0];
                fractions[0][2] *= newDenomVal2/fractions[1][2];
                int impnum2;
                
                if(fractions[2][2] == -1)impnum2 = fractions[0][0] + Math.abs(fractions[0][2]);
                else impnum2 = fractions[0][0] - fractions[0][2];
                
                fractions[0][3] = impnum2;
                fractions[1][3] = newDenomVal2;
                break;
                
           case "*":
                      
                int impnum3 = fractions[0][0] * fractions[0][2];                
                int denomVal3 = fractions[1][0] * fractions[1][2];
                
                fractions[0][3] = impnum3;
                fractions[1][3] = denomVal3;
                break;

                
           case "/":
           
                int impnum4 = fractions[0][0] * fractions[1][2];
                int denomVal4 = fractions[1][0] * fractions[0][2];
                
                fractions[0][3] = impnum4;
                fractions[1][3] = denomVal4;
                break;
                
        }

        //The resulting improper fraction is simplified and various 
        //special cases are accounted for.
        
        int num = fractions[0][3];
        num /= gcd(Math.abs(fractions[0][3]), Math.abs(fractions[1][3]));
        int denom = fractions[1][3];
        denom /= gcd(Math.abs(fractions[0][3]), Math.abs(fractions[1][3]));
        
        System.out.println(num + ", " + denom);

        if(num == 0 && denom == 0) return 0 + "";
        else if(num%denom == 0) return num/denom + "";
        else if(num/denom ==0) return num%denom + "/" + Math.abs(denom);
        else if(num==0) return 0 + "";
        else return num/denom + "_" + Math.abs(num)%Math.abs(denom) + "/" + Math.abs(denom);
        

        
        //return "whole: " + whole + " numerator: " + numer + " denominator: " + denom;
        
    }
}
    // TODO: Fill in the space below with any helper methods that you think you will need


