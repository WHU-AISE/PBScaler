import org.openqa.selenium.By;
import org.openqa.selenium.JavascriptExecutor;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.util.List;
import java.util.concurrent.TimeUnit;

public class TestServicePrice {
    private WebDriver driver;
    private String baseUrl;
    @BeforeClass
    public void setUp() throws Exception {
        System.setProperty("webdriver.chrome.driver", "D:/Program/chromedriver_win32/chromedriver.exe");
        driver = new ChromeDriver();
        baseUrl = "http://10.141.212.24/";
        driver.manage().timeouts().implicitlyWait(30, TimeUnit.SECONDS);
    }
    @Test
    public void testPrice() throws Exception{
        driver.get(baseUrl + "/");
        JavascriptExecutor js = (JavascriptExecutor) driver;
        js.executeScript("document.getElementById('price_update_startingPlace').value='shanghai'");
        js.executeScript("document.getElementById('price_update_endPlace').value='beijing'");
        js.executeScript("document.getElementById('price_update_distance').value='300'");

        driver.findElement(By.id("price_update_button")).click();
        Thread.sleep(1000);
//        String statusSignIn = driver.findElement(By.id("login_result_msg")).getText();
//        if(statusSignIn ==null || statusSignIn.length() <= 0) {
//            System.out.println("Failed,Status of Sign In btn is NULL!");
//            driver.quit();
//        }else
//            System.out.println("Sign Up btn status:"+statusSignIn);
    }
    @Test (dependsOnMethods = {"testPrice"})
    public void testQueryPrice() throws Exception{
        driver.findElement(By.id("price_queryAll_button")).click();
        Thread.sleep(1000);
        //gain Travel list
        List<WebElement> priceList = driver.findElements(By.xpath("//table[@id='query_price_list_table']/tbody/tr"));
        if (priceList.size() > 0)
            System.out.printf("Success to Query Price and Price list size is %d.%n",priceList.size());
        else
            System.out.println("Failed to Query Price or Price list size is 0");
        Assert.assertEquals(priceList.size() > 0,true);
    }
    @AfterClass
    public void tearDown() throws Exception {
        driver.quit();
    }
}
