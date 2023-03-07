import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.util.List;
import java.util.concurrent.TimeUnit;


public class TestServiceSecurity {
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
    public void testSecurity() throws Exception {
        driver.get(baseUrl + "/");
        driver.findElement(By.id("refresh_security_config_button")).click();
        Thread.sleep(1000);
        List<WebElement> securityList = driver.findElements(By.xpath("//table[@id='security_config_list_table']/tbody/tr"));
        if(securityList.size() > 0) {
            System.out.printf("Success,Security Config List's size is %d.%n", securityList.size());
            testSecurityCheck();
        }
        else
            System.out.println("False,Security Config List's size is 0 or Failed");
        Assert.assertEquals(securityList.size() > 0,true);
    }
    public void testSecurityCheck() throws Exception{
        driver.findElement(By.id("security_check_account_id")).clear();
        driver.findElement(By.id("security_check_account_id")).sendKeys("4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f");
        driver.findElement(By.id("security_check_button")).click();
        Thread.sleep(1000);
        String statusSecurityCheck = driver.findElement(By.id("security_check_message")).getText();
        if (!"".equals(statusSecurityCheck))
            System.out.println("Success: "+statusSecurityCheck);
        else
            System.out.println("False, status security check is null!");
        Assert.assertEquals(statusSecurityCheck.startsWith("Success"),true);
    }
    @AfterClass
    public void tearDown() throws Exception {
        driver.quit();
    }
}
